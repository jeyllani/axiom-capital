import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.abspath('src'))

from portfolio_engine.data_loader import PortfolioDataLoader
from portfolio_engine.optimizers import MinVarianceOptimizer, MaxSharpeOptimizer, RiskParityOptimizer, ResampledOptimizer
from portfolio_engine.weights_management import OptimizedWeightUpdater
from portfolio_engine.utils import get_optimal_solver
import riskfolio as rp
import plotly.graph_objects as go
import plotly.express as px
from portfolio_engine.technicals import get_technical_indicators

# st.set_page_config removed (handled by app.py)

# Custom CSS for Metric Cards
st.markdown("""
<style>
    [data-testid="stSidebar"] {display: block !important;}

    .metric-card {
        background-color: #1F2937; /* Dark Gray */
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #374151;
        margin-bottom: 1rem;
    }
    h5 {color: #9CA3AF; margin-bottom: 0.5rem; font-size: 0.9rem;} /* Light Gray */
    h2 {color: #F3F4F6; margin: 0; font-size: 1.8rem;} /* White */
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def load_data(config):
    """Loads data using PortfolioDataLoader and caches it in session state."""
    loader = PortfolioDataLoader(config)
    loader.load_data()
    return loader

def calculate_market_benchmark(loader):
    """
    Calculates the Value-Weighted (Market Cap) Benchmark Return.
    """
    returns, caps, _, _ = loader.get_matrices()
    
    if caps is None or returns is None:
        return None
        
    # Align shapes
    common_index = returns.index.intersection(caps.index)
    returns = returns.loc[common_index]
    caps = caps.loc[common_index]
    
    # Lag Caps
    lagged_caps = caps.shift(1)
    
    # Weights
    weights = lagged_caps.div(lagged_caps.sum(axis=1), axis=0)
    
    # VW Return
    vw_ret = (weights * returns).sum(axis=1)
    
    # Fallback for first period
    if pd.isna(vw_ret.iloc[0]):
        w0 = caps.iloc[0] / caps.iloc[0].sum()
        vw_ret.iloc[0] = (w0 * returns.iloc[0]).sum()
        
    return (1 + vw_ret).cumprod()

def calculate_ew_benchmark(loader):
    """
    Calculates the Equal-Weighted Benchmark Return.
    """
    returns, _, _, _ = loader.get_matrices()
    if returns is None: return None
    
    # EW Return = Mean of returns across assets
    ew_ret = returns.mean(axis=1)
    return (1 + ew_ret).cumprod()

def display_portfolio_analysis(results, loader, title="Portfolio Analysis", benchmark_results=None, market_benchmark=None, ew_benchmark=None, solver="N/A", cov_method="N/A", risk_aversion="N/A", max_turnover="N/A", cost_bps="N/A"):
    """
    Displays portfolio analysis with 'Conservative' design.
    Refined Charts: Solid lines, no fill, multiple benchmarks (Base, VW, EW).
    """
    values = results['values']
    weights_history = results['weights']
    dates = results['dates']
    rf_series = results.get('rf_series', None)
    
    # --- Metrics Calculation ---
    def calc_metrics(vals, rf_ser=None):
        rets = vals.pct_change().dropna()
        cum = vals.iloc[-1]
        ann_r = rets.mean() * 12
        ann_v = rets.std() * np.sqrt(12)
        
        if rf_ser is not None:
            rf_aligned = rf_ser.reindex(rets.index).fillna(0)
            excess = rets - rf_aligned
            shp = (excess.mean() * 12) / ann_v if ann_v > 0 else 0
        else:
            shp = (ann_r - 0.02) / ann_v if ann_v > 0 else 0
            
        dd = calculate_drawdown(vals).min()
        return cum, ann_r, ann_v, shp, dd

    cum_ret, ann_ret, ann_vol, sharpe, max_drawdown = calc_metrics(values, rf_series)
    turnover = results['turnover']
    
    # Benchmark Metrics (Base Strategy)
    b_cum, b_ret, b_vol, b_sharpe, b_dd = None, None, None, None, None
    if benchmark_results:
        b_cum, b_ret, b_vol, b_sharpe, b_dd = calc_metrics(benchmark_results['values'], benchmark_results.get('rf_series', None))
    
    # --- Metric Cards ---
    m1, m2, m3, m4, m5 = st.columns(5)
    
    def metric_card(label, value, delta_val=None, delta_text=None):
        delta_html = ""
        if delta_val is not None:
            color = "#10B981" if delta_val >= 0 else "#EF4444"
            delta_html = f'<div style="color: {color}; font-size: 0.9rem; margin-top: 4px;">{delta_text}</div>'
            
        return f"""
        <div class="metric-card">
            <h5>{label}</h5>
            <h2>{value}</h2>
            {delta_html}
        </div>
        """
    
    with m1:
        delta_val = (ann_ret - b_ret) if b_ret is not None else None
        delta_str = f"vs Base: {delta_val:+.2%}" if delta_val is not None else ""
        st.markdown(metric_card("Ann. Return", f"{ann_ret:.2%}", delta_val, delta_str), unsafe_allow_html=True)

    with m2:
        delta_val = (ann_vol - b_vol) if b_vol is not None else None
        delta_str = f"vs Base: {delta_val:+.2%}" if delta_val is not None else ""
        st.markdown(metric_card("Ann. Volatility", f"{ann_vol:.2%}", -delta_val if delta_val is not None else None, delta_str), unsafe_allow_html=True)

    with m3:
        delta_val = (sharpe - b_sharpe) if b_sharpe is not None else None
        delta_str = f"vs Base: {delta_val:+.2f}" if delta_val is not None else ""
        st.markdown(metric_card("Sharpe Ratio", f"{sharpe:.2f}", delta_val, delta_str), unsafe_allow_html=True)

    with m4:
        delta_val = (max_drawdown - b_dd) if b_dd is not None else None
        delta_str = f"vs Base: {delta_val:+.2%}" if delta_val is not None else ""
        st.markdown(metric_card("Max Drawdown", f"{max_drawdown:.2%}", delta_val, delta_str), unsafe_allow_html=True)

    with m5:
        # Turnover is monthly average
        st.markdown(metric_card("Avg. Turnover", f"{turnover:.2%}", None, ""), unsafe_allow_html=True)
        
    # --- CSS Injection for Blue Tabs ---
    st.markdown("""
    <style>
    /* Tab Selection & Hover - Blue Theme */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: transparent !important;
        border-bottom: 2px solid #3B82F6 !important; /* Blue-500 */
        color: #3B82F6 !important;
    }
    .stTabs [data-baseweb="tab-list"] button:hover {
        color: #60A5FA !important; /* Blue-400 */
        border-bottom-color: #60A5FA !important;
    }
    </style>
    """, unsafe_allow_html=True)
        
    if benchmark_results is None and title == "Resampled Optimization (Michaud)":
        st.warning("⚠️ Run 'Base Optimization' to see comparison metrics and charts.")
        
    st.markdown("---")
    
    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Performance", "🥧 Allocation", "⚡ Technicals", "📋 Details", "📥 Export"])
    
    with tab1:
        st.subheader("Cumulative Performance")
        
        # Stacked Chart (Price + Drawdown)
        from plotly.subplots import make_subplots
        
        cum_ret_series = values * 100 # Base 100
        drawdown_series = calculate_drawdown(values)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.7, 0.3],
                            subplot_titles=("Growth of $100", "Drawdown"))
        
        # Trace 1: Portfolio (Blue, Solid)
        fig.add_trace(go.Scatter(x=cum_ret_series.index, y=cum_ret_series, name='Strategy', 
                                 line=dict(color='#22D3EE', width=2.5)), row=1, col=1)
        
        # Trace 1b: Base Strategy (Gray, Solid) - If Resampling
        if benchmark_results:
            b_series = benchmark_results['values'] * 100
            fig.add_trace(go.Scatter(x=b_series.index, y=b_series, name='Base Strategy',
                                     line=dict(color='#9CA3AF', width=2)), row=1, col=1)
                                     
        # Trace 1c: Market VW (Orange, Solid)
        if market_benchmark is not None:
            mkt_series = market_benchmark.reindex(cum_ret_series.index).ffill() * 100
            fig.add_trace(go.Scatter(x=mkt_series.index, y=mkt_series, name='Market (VW)',
                                     line=dict(color='#F59E0B', width=2)), row=1, col=1)

        # Trace 1d: Market EW (Purple, Solid)
        if ew_benchmark is not None:
            ew_series = ew_benchmark.reindex(cum_ret_series.index).ffill() * 100
            fig.add_trace(go.Scatter(x=ew_series.index, y=ew_series, name='Market (EW)',
                                     line=dict(color='#A78BFA', width=2)), row=1, col=1)
        
        # Trace 2: Drawdown (Red Area)
        fig.add_trace(go.Scatter(x=drawdown_series.index, y=drawdown_series, name='Drawdown',
                                 line=dict(color='#EF4444', width=1), fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.2)'), row=2, col=1)
        
        # Annotation for Max Drawdown
        min_dd_idx = drawdown_series.idxmin()
        min_dd_val = drawdown_series.min()
        
        fig.add_annotation(
            x=min_dd_idx,
            y=min_dd_val,
            text=f"Max DD: {min_dd_val:.1%}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#3B82F6", # Blue Arrow
            ax=0,
            ay=40,
            row=2, col=1,
            font=dict(color="#3B82F6")
        )
        
        if benchmark_results:
            b_dd_series = calculate_drawdown(benchmark_results['values'])
            fig.add_trace(go.Scatter(x=b_dd_series.index, y=b_dd_series, name='Base Drawdown',
                                     line=dict(color='#6B7280', width=1)), row=2, col=1)
        
        fig.update_layout(height=600, showlegend=True, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified", template="plotly_dark")
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True, key=f"chart_cum_perf_{title}")
        
        # Monthly Returns
        st.subheader("Monthly Returns")
        returns = values.pct_change().dropna()
        monthly_rets = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        colors = ['#10B981' if r > 0 else '#EF4444' for r in monthly_rets]
        
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=monthly_rets.index, y=monthly_rets, marker_color=colors, name='Return'))
        fig_bar.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), template="plotly_dark", showlegend=False)
        fig_bar.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig_bar, use_container_width=True, key=f"chart_monthly_{title}")

    with tab2:
        # Final Allocation
        if weights_history:
            final_weights = weights_history[-1]
            
            col_a1, col_a2 = st.columns([1, 1])
            
            # Prepare Data
            w_df = final_weights.to_frame(name='Weight')
            
            # 1. Map to Raw Sectors
            raw_sectors = w_df.index.map(loader.sector_map).fillna('Other')
            
            # 2. Apply Unified Mapping (Safety Net)
            CRSP_MAPPING = {
                "Consumer Staples": "Basic Needs",
                "Consumer Defensive": "Basic Needs",
                "Consumer Discretionary": "Lifestyle & Luxury",
                "Consumer Cyclical": "Lifestyle & Luxury",
                "Financials": "Financial Services",
                "Information Technology": "Technology",
                "Materials": "Basic Materials",
                "Health Care": "Healthcare"
            }
            w_df['Sector'] = raw_sectors.map(lambda x: CRSP_MAPPING.get(x, x))
            w_df['Name'] = w_df.index.map(lambda x: loader.name_map.get(x, x) if loader.name_map else x)
            
            with col_a1:
                st.subheader("Sector Allocation")
                sector_alloc = w_df.groupby('Sector')['Weight'].sum().reset_index()
                fig_pie = px.pie(sector_alloc, values='Weight', names='Sector', title="Exposure by Sector", hole=0.4, color_discrete_sequence=px.colors.qualitative.Prism)
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(showlegend=False, template="plotly_dark")
                st.plotly_chart(fig_pie, use_container_width=True, key=f"chart_sector_{title}")
                
            with col_a2:
                st.subheader("Top 10 Holdings")
                top_10 = w_df.sort_values('Weight', ascending=False).head(10)
                fig_bar = px.bar(top_10, x='Weight', y='Name', orientation='h', title="Top Assets", text_auto='.1%')
                fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, template="plotly_dark")
                fig_bar.update_traces(marker_color='#22D3EE')
                st.plotly_chart(fig_bar, use_container_width=True, key=f"chart_top10_{title}")
                
            st.subheader("Allocation History (Wave Graph)")
            plot_sector_allocation(weights_history, dates, loader.sector_map, "Sector Allocation Over Time", key=f"chart_alloc_{title}")
            
            if benchmark_results:
                st.markdown("---")
                st.subheader("Base Strategy Allocation (Comparison)")
                plot_sector_allocation(benchmark_results['weights'], benchmark_results['dates'], loader.sector_map, "Base Strategy Allocation", key=f"chart_alloc_base_{title}")

            # Full Asset List (New)
            st.markdown("---")
            st.subheader("Full Asset List")
            
            full_list = w_df.copy()
            full_list['Ticker'] = full_list.index
            full_list = full_list[['Ticker', 'Name', 'Sector', 'Weight']]
            full_list = full_list.sort_values('Weight', ascending=False)
            
            # Reset Index to start from 1
            full_list.index = range(1, len(full_list) + 1)
            
            st.dataframe(
                full_list.style.format({'Weight': '{:.2%}'}),
                use_container_width=True
            )

    with tab3:
        st.subheader("Technical Indicators")
        df_tech = get_technical_indicators(values)
        
        # RSI
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df_tech.index, y=df_tech['RSI'], line=dict(color='#A78BFA', width=2), name='RSI'))
        fig_rsi.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dot", line_color="green", annotation_text="Oversold")
        fig_rsi.update_layout(title="Relative Strength Index (RSI)", height=250, template="plotly_dark", yaxis_range=[0, 100], margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_rsi, use_container_width=True, key=f"chart_rsi_{title}")
        
        # MACD
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MACD'], line=dict(color='#60A5FA', width=2), name='MACD'))
        fig_macd.add_trace(go.Scatter(x=df_tech.index, y=df_tech['Signal'], line=dict(color='#F472B6', width=2), name='Signal'))
        fig_macd.add_trace(go.Bar(x=df_tech.index, y=df_tech['MACD'] - df_tech['Signal'], marker_color='gray', name='Hist'))
        fig_macd.update_layout(title="MACD", height=250, template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_macd, use_container_width=True, key=f"chart_macd_{title}")

        # Stochastic
        fig_stoch = go.Figure()
        fig_stoch.add_trace(go.Scatter(x=df_tech.index, y=df_tech['%K'], line=dict(color='#34D399', width=2), name='%K'))
        fig_stoch.add_trace(go.Scatter(x=df_tech.index, y=df_tech['%D'], line=dict(color='#FBBF24', width=2), name='%D'))
        fig_stoch.add_hline(y=80, line_dash="dot", line_color="red", annotation_text="Overbought")
        fig_stoch.add_hline(y=20, line_dash="dot", line_color="green", annotation_text="Oversold")
        fig_stoch.update_layout(title="Stochastic Oscillator", height=250, template="plotly_dark", yaxis_range=[0, 100], margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_stoch, use_container_width=True, key=f"chart_stoch_{title}")

    with tab4:
        st.subheader("Comprehensive Strategy Report")
        
        # --- Advanced Metrics Calculation ---
        returns = values.pct_change().dropna()
        
        # Risk Metrics
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        skew = returns.skew()
        kurt = returns.kurtosis()
        
        # Ratios
        sortino = (ann_ret - 0.02) / (returns[returns < 0].std() * np.sqrt(12)) if returns[returns < 0].std() > 0 else 0
        calmar = ann_ret / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Stats
        win_rate = (returns > 0).mean()
        best_month = returns.max()
        worst_month = returns.min()
        
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            st.markdown("### 🛠️ Configuration & Inputs")
            cfg = loader.config
            
            config_data = {
                "Parameter": [
                    "Data Source", "Universe Size", "Date Range", "Lookback Period",
                    "Selection Method", "Risk Aversion (λ)", "Max Turnover (Constraint)", "Avg. Realized Turnover", "Transaction Cost",
                    "Risk-Free Rate", "Covariance Method", "Solver", "Max Drawdown Date"
                ],
                "Value": [
                    cfg.get('source', 'N/A'),
                    f"{cfg.get('n_stocks', 'N/A')} Assets",
                    f"{cfg.get('test_start')} to {cfg.get('test_end')}",
                    f"{cfg.get('lookback_months', 24)} Months",
                    cfg.get('selection_method', 'N/A').replace('_', ' ').title(),
                    f"{risk_aversion}", 
                    f"{max_turnover}",
                    f"{turnover:.2%}",
                    f"{cost_bps*10000:.0f} bps" if isinstance(cost_bps, float) else f"{cost_bps}",
                    "Dynamic (FF3)" if use_dynamic_rf else "Fixed (0%)",
                    cov_method,
                    solver,
                    f"{drawdown_series.idxmin().date()}"
                ]
            }
            st.table(pd.DataFrame(config_data).set_index("Parameter"))
            
            if 'sector_constraints' in st.session_state and st.session_state.sector_constraints:
                st.markdown("**Sector Constraints:**")
                st.json(st.session_state.sector_constraints)

        with col_d2:
            st.markdown("### 📊 Advanced Performance Metrics")
            
            perf_data = {
                "Metric": [
                    "Cumulative Return", "Annualized Return", "Annualized Volatility",
                    "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
                    "Max Drawdown", "Value at Risk (95%)", "CVaR (95%)",
                    "Skewness", "Kurtosis", "Win Rate (Months)",
                    "Best Month", "Worst Month", "Avg Turnover"
                ],
                "Value": [
                    f"{cum_ret:.2f}x", f"{ann_ret:.2%}", f"{ann_vol:.2%}",
                    f"{sharpe:.2f}", f"{sortino:.2f}", f"{calmar:.2f}",
                    f"{max_drawdown:.2%}", f"{var_95:.2%}", f"{cvar_95:.2%}",
                    f"{skew:.2f}", f"{kurt:.2f}", f"{win_rate:.1%}",
                    f"{best_month:.2%}", f"{worst_month:.2%}", f"{turnover:.2%}"
                ]
            }
            st.table(pd.DataFrame(perf_data).set_index("Metric"))

            # --- CHATBOT CONTEXT INJECTION ---
            # Save these results so the bot can "see" them
            if weights_history:
                final_w = weights_history[-1]
                top_holdings = final_w.sort_values(ascending=False).head(5).to_dict()
                top_holdings_str = ", ".join([f"{k}: {v:.1%}" for k, v in top_holdings.items()])
                
                context_data = {
                    "Annualized Return": f"{ann_ret:.2%}",
                    "Volatility": f"{ann_vol:.2%}",
                    "Sharpe Ratio": f"{sharpe:.2f}",
                    "Max Drawdown": f"{max_drawdown:.2%}",
                    "Top Holdings": top_holdings_str,
                    "Turnover": f"{turnover:.2%}"
                }
                st.session_state['latest_optimization_context'] = context_data

    with tab5:
        st.subheader("Export Data")
        st.download_button(
            label="Download Portfolio Values (CSV)",
            data=values.to_csv(),
            file_name="portfolio_values.csv",
            mime="text/csv",
            key=f"dl_csv_{title}"
        )

def run_optimization(loader, risk_aversion, sector_constraints, max_turnover, cost_bps, solver, cov_method, use_dynamic_rf=False):
    """Runs the base optimization (no resampling) and returns results."""
    
    # 1. Prepare Data
    returns_df = loader.returns_matrix
    market_caps_df = loader.market_caps_matrix
    sector_map = loader.sector_map
    
    # Align dates
    test_start_dt = pd.to_datetime(loader.config['test_start'])
    test_end_dt = pd.to_datetime(loader.config['test_end'])
    
    # Filter for simulation period
    sim_returns = returns_df.loc[test_start_dt:test_end_dt]
    simulation_dates = sim_returns.index
    
    # Initialize Optimizer
    optimizer = MinVarianceOptimizer(
        risk_aversion=risk_aversion,
        max_turnover=max_turnover,
        solver=solver,
        cov_method=cov_method
    )
    
    # Initialize Weight Updater
    updater = OptimizedWeightUpdater()
    
    # Storage
    portfolio_values = [1.0]
    current_weights = None
    turnover_history = []
    weights_history = []
    
    # Progress Bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Walk-Forward Loop
    for t in range(len(simulation_dates)):
        date = simulation_dates[t]
        
        # Update Progress
        progress = (t + 1) / len(simulation_dates)
        progress_bar.progress(progress)
        status_text.text(f"Processing: {date.date()}")
        
        # Define Lookback Window
        window_start = date - pd.DateOffset(months=loader.config['lookback_months'])
        hist_returns = returns_df.loc[window_start:date].iloc[:-1] # Exclude current month
        
        if len(hist_returns) < 12:
            portfolio_values.append(portfolio_values[-1])
            continue
        
        # Optimize
        try:
            target_weights = optimizer.optimize(
                returns=hist_returns,
                prev_weights=current_weights,
                sector_constraints=sector_constraints,
                sector_map=sector_map
            )
        except Exception as e:
            # st.warning(f"Optimization failed at {date.date()}: {e}")
            target_weights = current_weights if current_weights is not None else pd.Series(1.0/len(hist_returns.columns), index=hist_returns.columns)
        
        # Update Weights & Calculate Return
        current_month_ret = sim_returns.loc[date]
        
        # Use 'monthly' mode to rebalance to target
        new_weights, turnover, gross_ret, net_ret = updater.update_weights(
            target_weights=target_weights,
            prev_weights=current_weights,
            mode='monthly',
            returns_month=current_month_ret,
            cost_bps=cost_bps
        )
        
        # Update State
        current_weights = new_weights
        portfolio_values.append(portfolio_values[-1] * (1 + net_ret))
        turnover_history.append(turnover)
        weights_history.append(new_weights)
        
    status_text.empty()
    progress_bar.empty()
    
    results = {
        'values': pd.Series(portfolio_values, index=[simulation_dates[0] - pd.DateOffset(days=1)] + list(simulation_dates)),
        'turnover': np.mean(turnover_history),
        'weights': weights_history,
        'dates': simulation_dates,
        'rf_series': loader.rf_series.reindex(sim_returns.index).fillna(0.0) if use_dynamic_rf and loader.rf_series is not None else None
    }
    
    return results

def run_resampling(loader, risk_aversion, sector_constraints, max_turnover, cost_bps, n_simulations, solver, cov_method, use_dynamic_rf=False):
    """Runs Michaud resampling and returns results."""
    
    # 1. Prepare Data (Same as base optimization)
    returns_df = loader.returns_matrix
    market_caps_df = loader.market_caps_matrix
    sector_map = loader.sector_map
    
    # Align dates
    test_start_dt = pd.to_datetime(loader.config['test_start'])
    test_end_dt = pd.to_datetime(loader.config['test_end'])
    
    # Filter for simulation period
    sim_returns = returns_df.loc[test_start_dt:test_end_dt]
    simulation_dates = sim_returns.index
    
    # Initialize Base Optimizer
    base_opt = MinVarianceOptimizer(
        risk_aversion=risk_aversion,
        max_turnover=max_turnover,
        solver=solver,
        cov_method=cov_method
    )
    
    # Initialize Resampled Optimizer
    optimizer = ResampledOptimizer(base_opt, n_simulations=n_simulations, seed=42)
    
    # Initialize Weight Updater
    updater = OptimizedWeightUpdater()
    
    # Storage
    portfolio_values = [1.0]
    current_weights = None
    turnover_history = []
    weights_history = []






    
    # Progress Bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Walk-Forward Loop
    for t in range(len(simulation_dates)):
        date = simulation_dates[t]
        
        # Update Progress
        progress = (t + 1) / len(simulation_dates)
        progress_bar.progress(progress)
        status_text.text(f"Resampling: {date.date()}")
        
        # Define Lookback Window
        window_start = date - pd.DateOffset(months=loader.config['lookback_months'])
        hist_returns = returns_df.loc[window_start:date].iloc[:-1]
        
        if len(hist_returns) < 12:
            portfolio_values.append(portfolio_values[-1])
            continue
        
        # Optimize (Resampled)
        try:
            target_weights = optimizer.optimize(
                returns=hist_returns,
                prev_weights=current_weights,
                sector_constraints=sector_constraints,
                sector_map=sector_map
            )
        except Exception as e:
            target_weights = current_weights if current_weights is not None else pd.Series(1.0/len(hist_returns.columns), index=hist_returns.columns)
        
        # Update Weights & Calculate Return
        current_month_ret = sim_returns.loc[date]
        
        new_weights, turnover, gross_ret, net_ret = updater.update_weights(
            target_weights=target_weights,
            prev_weights=current_weights,
            mode='monthly',
            returns_month=current_month_ret,
            cost_bps=cost_bps
        )
        
        # Update State
        current_weights = new_weights
        portfolio_values.append(portfolio_values[-1] * (1 + net_ret))
        turnover_history.append(turnover)
        weights_history.append(new_weights)
        
    status_text.empty()
    progress_bar.empty()
    
    results = {
        'values': pd.Series(portfolio_values, index=[simulation_dates[0] - pd.DateOffset(days=1)] + list(simulation_dates)),
        'turnover': np.mean(turnover_history),
        'weights': weights_history,
        'dates': simulation_dates,
        'rf_series': loader.rf_series.reindex(sim_returns.index).fillna(0.0) if use_dynamic_rf and loader.rf_series is not None else None
    }
    
    return results

def plot_sector_allocation(weights_history, dates, sector_map, title, key=None):
    """Helper to plot sector allocation wave graph."""
    if not weights_history:
        return
    
    sector_weights_over_time = []
    
    # Unified Mapping (Safety Net)
    CRSP_MAPPING = {
        "Consumer Staples": "Basic Needs",
        "Consumer Defensive": "Basic Needs",
        "Consumer Discretionary": "Lifestyle & Luxury",
        "Consumer Cyclical": "Lifestyle & Luxury",
        "Financials": "Financial Services",
        "Information Technology": "Technology",
        "Materials": "Basic Materials",
        "Health Care": "Healthcare"
    }

    for w in weights_history:
        # w is a Series of asset weights
        w_df = w.to_frame(name='weight')
        raw_sectors = w_df.index.map(sector_map).fillna('Other')
        w_df['sector'] = raw_sectors.map(lambda x: CRSP_MAPPING.get(x, x))
        
        sec_w = w_df.groupby('sector')['weight'].sum()
        sector_weights_over_time.append(sec_w)
    
    # Create DataFrame for plotting
    # Ensure all sectors are present in all rows (fill 0)
    all_sectors = list(set(sector_map.values()))
    if not sector_weights_over_time:
         st.warning("No sector weights to plot.")
         return

    df_plot = pd.DataFrame(sector_weights_over_time, index=dates).fillna(0)
    
    if df_plot.empty:
        st.warning("No sector data available for plotting.")
        return

    # Plot using Plotly for Stacked Area Chart (Wave Chart)
    fig = go.Figure()
    
    # Use Prism color palette
    colors = px.colors.qualitative.Prism
    
    # Add traces for each sector
    for i, sector in enumerate(df_plot.columns):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=df_plot.index, 
            y=df_plot[sector],
            mode='lines',
            name=sector,
            stackgroup='one', # This creates the stacked effect
            line=dict(width=0.5, color=color),
            fillcolor=color # Optional: adjust opacity if needed
        ))
        
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Allocation Weight",
        yaxis=dict(tickformat=".0%", range=[0, 1]), # Ensure 0-100% scale
        hovermode="x unified",
        template="plotly_dark",
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True, key=key)

def calculate_drawdown(series):
    """Calculates the drawdown series."""
    hwm = series.cummax()
    drawdown = (series / hwm) - 1
    return drawdown

def calculate_efficient_frontier(loader, points=50, solver=None):
    """
    Calculates the Efficient Frontier using Riskfolio-Lib.
    """
    # 1. Get Returns
    returns = loader.returns_matrix
    if returns is None or returns.empty:
        return None
        
    # Clean NaNs for Ledoit-Wolf (it doesn't support NaNs)
    returns = returns.fillna(0)
        
    # 2. Setup Portfolio
    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu='hist', method_cov='ledoit')
    
    # Set Solver
    if solver:
        port.solver = solver
    
    # 3. Calculate Frontier
    try:
        # points: number of points on the frontier
        ws = port.efficient_frontier(model='Classic', rm='MV', points=points, rf=0, hist=True)
        
        # 4. Get Risk/Return coordinates
        # Risk (Volatility)
        cov = port.cov
        mu = port.mu
        
        # Calculate Risk and Return for each point on the frontier
        frontier_vol = []
        frontier_ret = []
        
        for i in range(ws.shape[1]):
            w = ws.iloc[:, i].values
            ret = mu @ w
            vol = np.sqrt(w.T @ cov @ w)
            
            # Ensure scalars
            if isinstance(ret, (pd.Series, pd.DataFrame, np.ndarray)):
                ret = ret.item()
            if isinstance(vol, (pd.Series, pd.DataFrame, np.ndarray)):
                vol = vol.item()
                
            frontier_ret.append(ret * 12) # Annualize
            frontier_vol.append(vol * np.sqrt(12)) # Annualize
            
        # 5. Get Assets Risk/Return for scatter plot
        assets_vol = np.sqrt(np.diag(cov)) * np.sqrt(12)
        assets_ret = mu.values.flatten() * 12
        
        return {
            'frontier_vol': frontier_vol,
            'frontier_ret': frontier_ret,
            'assets_vol': assets_vol,
            'assets_ret': assets_ret,
            'assets_names': returns.columns.tolist()
        }
    except Exception as e:
        st.error(f"Efficient Frontier Calculation Failed: {e}")
        return None

# --- Sidebar ---
with st.sidebar:
    # Hide default navigation & Reduce Margins
    st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
        .block-container {padding-top: 4rem; padding-bottom: 0rem;}
        section[data-testid="stSidebar"] {padding-top: 0rem;}
        div[data-testid="stSidebarUserContent"] {padding-top: 1rem;}
        
        /* Back Button Styling (Sidebar) */
        section[data-testid="stSidebar"] .stButton button {
            width: auto !important;
            padding: 8px 20px !important;
            background-color: transparent !important;
            border: 1px solid #475569 !important;
            color: #94a3b8 !important;
            border-radius: 20px !important;
            font-size: 14px !important;
            transition: all 0.2s ease;
            margin-bottom: 20px !important; /* Vertical Spacing */
        }
        section[data-testid="stSidebar"] .stButton button:hover {
            border-color: #cbd5e1 !important;
            color: #f8fafc !important;
            background-color: rgba(255, 255, 255, 0.05) !important;
        }
    </style>
    """, unsafe_allow_html=True)

    if st.button("⬅️ Back to Expert Tools"):
        st.switch_page("pages/experts/landing_expert.py")
    
    # --- Data Configuration (Restored) ---
    st.header("1. Data Configuration")
    
    # Defaults from loaded data if available
    default_source = 0
    default_start = pd.to_datetime("2010-01-01")
    default_end = pd.to_datetime("2024-12-31")
    default_n_assets = 50
    default_method = 0 # Top Market Cap
    
    if st.session_state.loader is not None:
        cfg = st.session_state.loader.config
        if cfg['source'] == 'CRSP': default_source = 1
        default_start = pd.to_datetime(cfg['test_start'])
        default_end = pd.to_datetime(cfg['test_end'])
        default_n_assets = int(cfg['n_stocks'])
        if cfg.get('selection_method') == 'random': default_method = 1

    # Data Source (Static)
    st.caption("Data Source: YFinance")
    data_source = "YFINANCE"
    
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date", default_start, min_value=pd.to_datetime("2010-01-01"), max_value=pd.to_datetime("2024-12-31"))
    end_date = col2.date_input("End Date", default_end, min_value=pd.to_datetime("2010-01-01"), max_value=pd.to_datetime("2024-12-31"))
    
    st.subheader("Asset Selection")
    selection_method = st.radio("Method", ["Top Market Cap", "Random Selection"], index=default_method)
    n_assets = st.slider("Number of Assets", min_value=10, max_value=100, value=default_n_assets, step=10)

    # --- Constraints ---
    st.markdown("---")
    st.markdown("---")
    st.header("2. Constraints")
    enable_sector_constraints = st.checkbox("Enable Sector Constraints")
    
    sector_constraints = None
    if enable_sector_constraints:
        with st.expander("Sector Allocation Limits", expanded=True):
            # Unified Sector List (Target Names)
            UNIFIED_SECTORS = [
                "Basic Needs",          # ex-Consumer Staples / Defensive
                "Lifestyle & Luxury",   # ex-Consumer Discretionary / Cyclical
                "Financial Services",   # ex-Financials
                "Technology",           # ex-Information Technology
                "Healthcare",
                "Energy",
                "Industrials",
                "Basic Materials",      # ex-Materials
                "Communication Services",
                "Real Estate",
                "Utilities",
                "Fixed Income"
            ]
            
            # CRSP & Legacy YFinance to Unified Mapping
            CRSP_MAPPING = {
                "Consumer Staples": "Basic Needs",
                "Consumer Defensive": "Basic Needs", # Legacy YF
                "Consumer Discretionary": "Lifestyle & Luxury",
                "Consumer Cyclical": "Lifestyle & Luxury", # Legacy YF
                "Financials": "Financial Services",
                "Information Technology": "Technology",
                "Materials": "Basic Materials",
                "Health Care": "Healthcare" # Minor spelling diff
            }

            if 'loader' in st.session_state and st.session_state.loader is not None:
                loader = st.session_state.loader
                if loader.sector_map:
                    sector_constraints = {}
                    
                    # Apply Mapping to Loader's Sector Map (In-Memory Fix)
                    # This ensures the optimizer sees the Unified Names
                    remapped_map = {}
                    for ticker, sec in loader.sector_map.items():
                        # Remap if in CRSP mapping, else keep original
                        remapped_map[ticker] = CRSP_MAPPING.get(sec, sec)
                    
                    # Update the loader's map in session state so optimizer uses it
                    loader.sector_map = remapped_map
                    
                    # Display Sliders for Unified List
                    for sector in sorted(UNIFIED_SECTORS):
                        # Default 0-100%
                        min_w, max_w = st.slider(f"{sector}", 0.0, 1.0, (0.0, 1.0), key=f"sec_{sector}")
                        sector_constraints[sector] = (min_w, max_w)
                else:
                    st.warning("No sector data available. Load data first.")
            else:
                st.info("Load data to see sectors.")

    # Strategy Parameters
    st.markdown("---")
    st.markdown("---")
    st.header("3. Strategy Parameters")
    risk_aversion = st.slider("Risk Aversion (λ)", 1.0, 10.0, 5.0)
    max_turnover = st.slider("Max Monthly Turnover", 0.0, 1.0, 0.20)
    cost_bps = st.number_input("Transaction Cost (bps)", value=10) / 10000
    use_dynamic_rf = st.checkbox("Use Dynamic Risk-Free Rate (FRED)", value=False, help="Use historical Risk-Free Rate for Sharpe Ratio calculation.")

    # Advanced Method
    st.markdown("---")
    st.markdown("---")
    with st.expander("Advanced Method"):
        cov_method_display = st.selectbox("Covariance Method", ["Ledoit-Wolf", "Historic"], index=0)
        cov_method_map = {"Ledoit-Wolf": "ledoit", "Historic": "hist"}
        cov_method = cov_method_map[cov_method_display]
    # Solver Selection (Dynamic)
    # Solver Selection (Hardcoded)
    solver = "CLARABEL"

    st.markdown("---")
    st.header("4. Execution")
    # Run Optimization Button - Moved to bottom of sidebar
    run_btn = st.button("▶️ Run Optimization", type="primary", use_container_width=True)

# --- Main Area ---
st.title("Utility Maximization Strategy")

# Initialize Session State
if 'loader' not in st.session_state:
    st.session_state.loader = None
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None

# Construct Config from Sidebar Inputs
config = {
    'source': str(data_source),
    'data_dir': '../data/yfinance' if data_source == 'YFINANCE' else '../data/processed', # Adjust paths
    'input_file': 'financial_universe_clean.parquet' if data_source == 'YFINANCE' else 'returns_liquid_nyse80.parquet',
    'test_start': str(start_date),
    'test_end': str(end_date),
    'lookback_months': 24,
    'selection_method': 'top_market_cap' if selection_method == "Top Market Cap" else 'random',
    'n_stocks': int(n_assets),
    'rf_file': '../data/processed/risk_free_daily_2000_2025.parquet'
}

# Auto-Load / Reactive Reload Logic
should_reload = False
if st.session_state.loader is None:
    should_reload = True
else:
    # Check if critical parameters changed
    prev_config = st.session_state.loader.config
    
    # Robust comparison
    if (str(prev_config['source']) != config['source'] or
        str(prev_config['test_start']) != config['test_start'] or
        str(prev_config['test_end']) != config['test_end'] or
        int(prev_config['n_stocks']) != config['n_stocks'] or
        str(prev_config['selection_method']) != config['selection_method']):
        should_reload = True

if should_reload:
    with st.spinner("Updating Universe Data..."):
        try:
            st.session_state.loader = load_data(config)
            st.session_state.market_benchmark = calculate_market_benchmark(st.session_state.loader)
            st.session_state.ew_benchmark = calculate_ew_benchmark(st.session_state.loader)
            st.rerun()
        except Exception as e:
            st.error(f"Error loading data: {e}")

# Ensure Benchmarks are calculated (if missing)
if st.session_state.loader is not None:
    if 'market_benchmark' not in st.session_state or st.session_state.market_benchmark is None:
        st.session_state.market_benchmark = calculate_market_benchmark(st.session_state.loader)
    if 'ew_benchmark' not in st.session_state or st.session_state.ew_benchmark is None:
        st.session_state.ew_benchmark = calculate_ew_benchmark(st.session_state.loader)

# Handle Run Optimization Trigger
if run_btn:
    if st.session_state.loader is None:
        st.error("Please load data first.")
    else:
        with st.spinner("Optimizing..."):
            try:
                results = run_optimization(
                    st.session_state.loader,
                    risk_aversion,
                    sector_constraints,
                    max_turnover,
                    cost_bps,
                    solver,
                    cov_method,
                    use_dynamic_rf
                )
                st.session_state.optimization_results = results
                st.toast("Optimization Complete!", icon="✅")
            except Exception as e:
                st.error(f"Optimization failed: {e}")

# Tabs
tab_opt, tab_resamp = st.tabs(["📊 Optimization", "✨ Resampling"])

with tab_opt:
    st.markdown("### Base Optimization")
    
    # Display Results
    if st.session_state.optimization_results is not None:
        display_portfolio_analysis(
            st.session_state.optimization_results, 
            st.session_state.loader, 
            "Base Optimization",
            market_benchmark=st.session_state.get('market_benchmark'),
            ew_benchmark=st.session_state.get('ew_benchmark'),
            solver=solver,
            cov_method=cov_method,
            risk_aversion=risk_aversion,
            max_turnover=max_turnover,
            cost_bps=cost_bps
        )
    else:
        st.info("Click 'Run Optimization' in the sidebar to start.")
        


with tab_resamp:
    st.markdown("### Michaud Resampling")
    
    # Check if data is loaded
    if st.session_state.loader is None:
        st.info("Please load data first.")
    else:
        n_sims = st.slider("Number of Simulations", 1, 10, 1, key="n_sims_utility")
        
        if st.button("▶️ Run Resampling", key="btn_resamp_utility"):
            with st.spinner(f"Running {n_sims} simulations..."):
                # 1. Setup Resampled Optimizer (Note: The provided instruction defines this but doesn't use it in run_optimization)
                # base_opt = MinVarianceOptimizer(
                #     risk_aversion=risk_aversion,
                #     max_turnover=max_turnover,
                #     cov_method=cov_method,
                #     solver=solver
                # )
                # resamp_opt = ResampledOptimizer(base_opt, n_simulations=n_sims)
                
                # 2. Run Backtest (using the main run_optimization as per instruction)
                res_results = run_resampling( # Changed to run_resampling as per original code structure
                    loader=st.session_state.loader, # Use session state loader
                    risk_aversion=risk_aversion,
                    sector_constraints=sector_constraints,
                    max_turnover=max_turnover,
                    cost_bps=cost_bps,
                    n_simulations=n_sims, # Pass n_sims to run_resampling
                    solver=solver,
                    cov_method=cov_method,
                    use_dynamic_rf=use_dynamic_rf
                )
                
                # Store in Session State
                st.session_state['resamp_results'] = res_results
                st.success("Resampling Complete!")
        
        # Display Results
        if 'resamp_results' in st.session_state:
            display_portfolio_analysis(
                st.session_state['resamp_results'], 
                st.session_state.loader, 
                "Resampled Optimization (Michaud)",
                benchmark_results=st.session_state.optimization_results,
                market_benchmark=st.session_state.get('market_benchmark'),
                ew_benchmark=st.session_state.get('ew_benchmark'),
                solver=solver,
                cov_method=cov_method,
                risk_aversion=risk_aversion,
                max_turnover=max_turnover,
                cost_bps=cost_bps
            )


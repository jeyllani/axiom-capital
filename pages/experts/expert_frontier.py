import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import riskfolio as rp
import sys
import os

# Add src to path
sys.path.append(os.path.abspath('src'))

from portfolio_engine.data_loader import PortfolioDataLoader
from portfolio_engine.utils import get_optimal_solver

def load_data(config):
    """Helper to instantiate and load data."""
    loader = PortfolioDataLoader(config)
    loader.load_data()
    return loader
from portfolio_engine.optimizers import MinVarianceOptimizer, MaxSharpeOptimizer, TargetVolOptimizer
from portfolio_engine.utils import get_optimal_solver
from portfolio_engine.technicals import get_technical_indicators
from portfolio_engine.weights_management import OptimizedWeightUpdater

def calculate_drawdown(series):
    """Calculates drawdown series."""
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    return drawdown

# --- CONFIGURATION ---
# st.set_page_config removed (handled by app.py)

# Apply Dark Theme & Custom CSS (Standard)
st.markdown("""
<style>
    [data-testid="stSidebar"] {display: block !important;}

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
    
    /* Metric Cards */
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-card h5 {
        color: #94A3B8;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 8px;
    }
    .metric-card h2 {
        color: #F3F4F6;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

if st.sidebar.button("⬅️ Back to Expert Tools"):
    st.switch_page("pages/experts/landing_expert.py")


# --- HELPER FUNCTIONS (Standard Pattern) ---

def display_metric_card(title, value, prefix="", suffix="", color="#F3F4F6", delta_val=None, delta_text=None):
    delta_html = ""
    if delta_val is not None:
        color_delta = "#10B981" if delta_val >= 0 else "#EF4444"
        delta_html = f'<div style="color: {color_delta}; font-size: 0.9rem; margin-top: 4px;">{delta_text}</div>'
        
    st.markdown(f"""
    <div class="metric-card">
        <h5>{title}</h5>
        <h2 style="color: {color};">{prefix}{value}{suffix}</h2>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def plot_sector_allocation(weights_history, dates, sector_map, title, key=None):
    """Helper to plot sector allocation wave graph (Unified)."""
    if not weights_history:
        return
    
    sector_weights_over_time = []
    
    # Unified Mapping
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
        w_df = w.to_frame(name='weight')
        raw_sectors = w_df.index.map(sector_map).fillna('Other')
        w_df['sector'] = raw_sectors.map(lambda x: CRSP_MAPPING.get(x, x))
        sec_w = w_df.groupby('sector')['weight'].sum()
        sector_weights_over_time.append(sec_w)
    
    if not sector_weights_over_time: return

    # Ensure dates match length
    if len(sector_weights_over_time) != len(dates):
        # Slice dates to match weights (usually dates has one more if not aligned)
        dates = dates[:len(sector_weights_over_time)]

    df_plot = pd.DataFrame(sector_weights_over_time, index=dates).fillna(0)
    
    if df_plot.empty: return

    fig = go.Figure()
    colors = px.colors.qualitative.Prism
    
    for i, sector in enumerate(df_plot.columns):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot[sector], mode='lines', name=sector,
            stackgroup='one', line=dict(width=0.5, color=color), fillcolor=color
        ))
        
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Allocation",
        yaxis=dict(tickformat=".0%", range=[0, 1]), hovermode="x unified",
        template="plotly_dark", height=500, margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

def calculate_efficient_frontier(loader, points=50, solver=None):
    """Calculates Efficient Frontier using Riskfolio."""
    returns = loader.returns_matrix
    if returns is None or returns.empty: return None
    returns = returns.fillna(0)
    
    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu='hist', method_cov='ledoit')
    if solver: port.solver = solver
    
    try:
        ws = port.efficient_frontier(model='Classic', rm='MV', points=points, rf=0, hist=True)
        cov, mu = port.cov, port.mu
        
        frontier_vol, frontier_ret = [], []
        for i in range(ws.shape[1]):
            w = ws.iloc[:, i].values
            ret = mu @ w
            vol = np.sqrt(w.T @ cov @ w)
            frontier_ret.append(ret.item() * 12)
            frontier_vol.append(vol.item() * np.sqrt(12))
            
        assets_vol = np.sqrt(np.diag(cov)) * np.sqrt(12)
        assets_ret = mu.values.flatten() * 12
        
        return {
            'frontier_vol': frontier_vol, 'frontier_ret': frontier_ret,
            'assets_vol': assets_vol, 'assets_ret': assets_ret,
            'assets_names': returns.columns.tolist()
        }
    except Exception as e:
        st.error(f"Frontier Error: {e}")
        return None

def run_backtest(loader, optimizer_class, optimizer_params, sector_constraints, cost_bps, use_dynamic_rf=True, optimize_kwargs={}, benchmark_series=None, enable_resampling=False, n_simulations=100):
    """Runs walk-forward backtest."""
    returns = loader.returns_matrix
    if returns is None: return None
    
    dates = returns.index
    rebal_dates = dates[dates.is_month_end] # Monthly rebalancing
    
    # Instantiate Optimizer and Updater
    optimizer = optimizer_class(**optimizer_params)
    updater = OptimizedWeightUpdater() # Instantiate updater
    
    # Lookback Window
    lookback = loader.config.get('lookback_months', 24)    # Storage
    portfolio_values = [1.0]
    current_weights = None
    turnover_history = []
    weights_history = []
    
    # Initialize weights with 1/N for the start date (t=0)
    # This ensures weights_history has same length as simulation_dates
    initial_weights = pd.Series(1.0/len(returns.columns), index=returns.columns)
    weights_history.append(initial_weights)
    current_weights = initial_weights

    # Progress Bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Walk-Forward Loop
    # We iterate from t=1 to end, as t=0 is initial state
    for t in range(1, len(rebal_dates)):
        date = rebal_dates[t]
        
        # Update Progress
        progress = (t + 1) / len(rebal_dates)
        progress_bar.progress(progress)
        status_text.text(f"Optimizing: {date.date()}")
        
        # Define Lookback Window
        window_start = date - pd.DateOffset(months=loader.config['lookback_months'])
        hist_returns = returns.loc[window_start:date].iloc[:-1] # Exclude current month
        
        if len(hist_returns) < 12:
            # If not enough history, carry forward previous weights and value
            portfolio_values.append(portfolio_values[-1])
            weights_history.append(current_weights)
            turnover_history.append(0) # No turnover if weights are carried forward
            continue
        
        # Prepare Returns for Optimizer
        # If Benchmark is provided, we optimize on Excess Returns (Tracking Error minimization)
        opt_returns = hist_returns
        if benchmark_series is not None:
            # Align benchmark to history window
            bench_window = benchmark_series.reindex(hist_returns.index).fillna(0.0)
            opt_returns = hist_returns.sub(bench_window, axis=0)

        # Optimize
        try:
            # Resampling Logic
            if enable_resampling:
                sim_weights = []
                for _ in range(n_simulations):
                    resampled_rets = hist_returns.sample(n=len(hist_returns), replace=True)
                    w = optimizer.optimize(
                        resampled_rets, 
                        prev_weights=current_weights, 
                        sector_constraints=sector_constraints,
                        sector_map=loader.sector_map,
                        **optimize_kwargs
                    )
                    sim_weights.append(w)
                new_weights = pd.DataFrame(sim_weights).mean()
                new_weights = new_weights / new_weights.sum()
            else:
                new_weights = optimizer.optimize(
                    hist_returns, 
                    prev_weights=current_weights, 
                    sector_constraints=sector_constraints,
                    sector_map=loader.sector_map,
                    **optimize_kwargs
                )
            target_weights = new_weights
        except Exception as e:
            # Fallback to current weights or Equal Weight
            st.warning(f"Optimization failed for {date.date()}: {e}. Carrying forward previous weights.")
            target_weights = current_weights if current_weights is not None else pd.Series(1.0/len(hist_returns.columns), index=hist_returns.columns)
        
        # Update Weights & Calculate Return
        # The return for the current month (date) is used to calculate the portfolio value
        # based on the weights from the *previous* rebalance.
        # The new weights (target_weights) will be applied for the *next* month.
        
        # Get the returns for the month *after* the optimization period (i.e., the month we are currently in 'date')
        # This assumes 'date' is the end of the month for which we are calculating returns.
        # The weights optimized using data up to 'date' will be applied for the period *after* 'date'.
        # For simplicity in this backtest, we'll use the returns of the month 'date' with the *previous* weights
        # and then apply the new weights for the *next* period.
        
        # Let's assume `rebal_dates` are month-ends.
        # `returns.loc[date]` gives the returns for the month ending on `date`.
        # We optimize using data up to `date-1`. The new weights are for the month `date`.
        
        # Calculate the return for the current month using the *previous* weights
        # This is a common simplification in backtests where rebalancing happens at month-end.
        # The weights determined at the end of month T-1 are applied for month T.
        # Here, `current_weights` are the weights from the *previous* rebalance.
        # `returns.loc[date]` are the returns for the month ending `date`.
        
        # The `updater` handles the transition.
        # `current_weights` are the weights *before* rebalancing at `date`.
        # `target_weights` are the weights *after* rebalancing at `date`.
        # `returns.loc[date]` are the returns for the period *after* the previous rebalance and *before* the current rebalance.
        
        # The `PortfolioUpdater` expects `returns_month` to be the returns for the period
        # *after* `prev_weights` were set and *before* `target_weights` are applied.
        # So, `returns.loc[date]` is correct here.
        
        new_weights, turnover, gross_ret, net_ret = updater.update_weights(
            target_weights=target_weights,
            prev_weights=current_weights,
            mode='monthly',
            returns_month=returns.loc[date],
            cost_bps=cost_bps
        )
        
        # Update State
        current_weights = new_weights
        portfolio_values.append(portfolio_values[-1] * (1 + net_ret))
        turnover_history.append(turnover)
        weights_history.append(new_weights)
        
    status_text.empty()
    progress_bar.empty()
    
    # Correctly construct RF Series aligned with simulation dates
    rf_aligned_series = None
    if use_dynamic_rf and loader.rf_series is not None:
        # Align RF series to the dates for which portfolio values are calculated
        rf_aligned_series = loader.rf_series.reindex(rebal_dates, method='ffill').fillna(0.0)

    results = {
        'values': pd.Series(portfolio_values, index=rebal_dates),
        'turnover': np.mean(turnover_history) if turnover_history else 0.0,
        'weights': weights_history,
        'dates': rebal_dates, # These are the rebalancing dates
        'rf_series': rf_aligned_series
    }
    
    return results

def display_frontier_results(results, loader, title, benchmark_results=None):
    """Displays results in standard format with Benchmarks and Full Tabs."""
    values = results['values']
    weights_history = results['weights']
    dates = results['dates']
    
    # --- 1. Calculate Benchmarks (EW Only) ---
    returns_matrix = loader.returns_matrix
    
    # Align to strategy dates
    strat_start = values.index[0]
    strat_end = values.index[-1]
    
    # EW Benchmark
    ew_index = returns_matrix.loc[strat_start:strat_end].mean(axis=1)
    ew_values = (1 + ew_index).cumprod()
    ew_values = ew_values / ew_values.iloc[0] # Normalize to 1
    
    # VW Benchmark
    caps_matrix = loader.market_caps_matrix
    vw_values = None
    if caps_matrix is not None and not caps_matrix.empty:
        caps_aligned = caps_matrix.reindex(returns_matrix.index).reindex(returns_matrix.columns, axis=1).fillna(0)
        total_cap = caps_aligned.sum(axis=1).replace(0, np.nan)
        weighted_rets = (returns_matrix * caps_aligned).sum(axis=1) / total_cap
        vw_index = weighted_rets.loc[strat_start:strat_end].fillna(0)
        vw_values = (1 + vw_index).cumprod()
        vw_values = vw_values / vw_values.iloc[0] # Normalize
    
    # --- 2. Metrics Calculation ---
    rets = values.pct_change().dropna()
    
    # Get RF rate for Sharpe calculation
    rf_rate = 0.0
    if 'rf_series' in results and results['rf_series'] is not None:
        aligned_rf = results['rf_series'].reindex(rets.index, method='ffill').fillna(0.0)
        rf_rate = aligned_rf.mean()
    
    ann_ret = rets.mean() * 12
    ann_vol = rets.std() * np.sqrt(12)
    sharpe = (ann_ret - (rf_rate * 12)) / ann_vol if ann_vol > 0 else 0
    max_drawdown = calculate_drawdown(values).min()
    turnover = results.get('turnover', 0.0)
    
    # --- Context Injection for Chatbot ---
    try:
        top_holdings_str = "N/A"
        if weights_history:
            last_w = weights_history[-1]
            top5 = last_w.sort_values(ascending=False).head(5)
            # Map names if possible
            top5_list = []
            for t, w in top5.items():
                name = loader.name_map.get(t, t) if loader.name_map else t
                top5_list.append(f"{name} ({w:.1%})")
            top_holdings_str = ", ".join(top5_list)

        context_data = {
            "Page": "Expert: Efficient Frontier",
            "Annual Return": f"{ann_ret:.2%}",
            "Volatility": f"{ann_vol:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Top Holdings": top_holdings_str
        }
        st.session_state['latest_optimization_context'] = context_data
    except Exception as e:
        print(f"Context Injection Error: {e}")
    
    # Benchmark Metrics (Base Strategy if Resampling, else EW)
    if benchmark_results:
        bench_vals = benchmark_results['values']
        bench_name = "Initial Strategy"
    else:
        bench_vals = ew_values
        bench_name = "EW Benchmark"
    
    b_rets = bench_vals.pct_change().dropna()
    b_ann_ret = b_rets.mean() * 12
    b_ann_vol = b_rets.std() * np.sqrt(12)
    b_sharpe = (b_ann_ret - (rf_rate * 12)) / b_ann_vol if b_ann_vol > 0 else 0
    b_dd = calculate_drawdown(bench_vals).min()

    # Metric Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    
    with m1:
        d_val = (ann_ret - b_ann_ret) if benchmark_results else None
        d_txt = f"vs {bench_name}: {d_val:+.2%}" if d_val is not None else ""
        display_metric_card("Ann. Return", f"{ann_ret:.2%}", delta_val=d_val, delta_text=d_txt)
    with m2:
        d_val = (ann_vol - b_ann_vol) if benchmark_results else None
        d_txt = f"vs {bench_name}: {d_val:+.2%}" if d_val is not None else ""
        display_metric_card("Ann. Volatility", f"{ann_vol:.2%}", delta_val=-d_val if d_val is not None else None, delta_text=d_txt)
    with m3:
        d_val = (sharpe - b_sharpe) if benchmark_results else None
        d_txt = f"vs {bench_name}: {d_val:+.2f}" if d_val is not None else ""
        display_metric_card("Sharpe Ratio", f"{sharpe:.2f}", delta_val=d_val, delta_text=d_txt)
    with m4:
        d_val = (max_drawdown - b_dd) if benchmark_results else None
        d_txt = f"vs {bench_name}: {d_val:+.2%}" if d_val is not None else ""
        display_metric_card("Max Drawdown", f"{max_drawdown:.2%}", delta_val=d_val, delta_text=d_txt)
    with m5:
        display_metric_card("Avg. Turnover", f"{turnover:.2%}")

    # --- 3. Tabs ---
    t_perf, t_alloc, t_tech, t_det, t_exp = st.tabs(["📈 Performance", "🥧 Allocation", "⚡ Technicals", "📋 Details", "📥 Export"])
    
    with t_perf:
        st.subheader("Cumulative Performance")
        
        # Stacked Chart (Price + Drawdown)
        cum_ret_series = values * 100 # Base 100
        drawdown_series = calculate_drawdown(values)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.7, 0.3],
                            subplot_titles=("Growth of $100", "Drawdown"))
        
        # Trace 1: Portfolio
        fig.add_trace(go.Scatter(x=cum_ret_series.index, y=cum_ret_series, name='Strategy', 
                                 line=dict(color='#22D3EE', width=2.5)), row=1, col=1)
        
        # Trace 1b: Benchmark (Initial)
        if benchmark_results:
            b_series = benchmark_results['values'] * 100
            b_series = b_series / b_series.iloc[0] * 100
            fig.add_trace(go.Scatter(x=b_series.index, y=b_series, name="Initial Strategy",
                                     line=dict(color='#9CA3AF', width=2, dash='dot')), row=1, col=1)
        
        # Trace 1c: VW Benchmark
        if vw_values is not None:
            fig.add_trace(go.Scatter(x=vw_values.index, y=vw_values*100, name='Market Cap Weighted (VW)',
                                     line=dict(color='#F59E0B', width=2, dash='dot')), row=1, col=1)
        
        # Trace 1d: EW Benchmark
        fig.add_trace(go.Scatter(x=ew_values.index, y=ew_values*100, name='Equal Weighted (EW)',
                                 line=dict(color='#A78BFA', width=2, dash='dot')), row=1, col=1)

        # Trace 2: Drawdown
        fig.add_trace(go.Scatter(x=drawdown_series.index, y=drawdown_series, name='Drawdown',
                                 line=dict(color='#EF4444', width=1), fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.2)'), row=2, col=1)
        
        # Trace 2b: Benchmark Drawdown
        b_dd_series = calculate_drawdown(bench_vals)
        fig.add_trace(go.Scatter(x=b_dd_series.index, y=b_dd_series, name=f'{bench_name} Drawdown',
                                 line=dict(color='#6B7280', width=1)), row=2, col=1)

        fig.update_layout(height=600, showlegend=True, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True, key=f"chart_cum_perf_{title}")
        
        # Monthly Returns
        st.subheader("Monthly Returns")
        monthly_rets = rets.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        colors = ['#10B981' if r > 0 else '#EF4444' for r in monthly_rets]
        
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=monthly_rets.index, y=monthly_rets, marker_color=colors, name='Monthly Return'))
        fig_bar.update_layout(title="Monthly Returns Distribution", template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_bar, use_container_width=True, key=f"chart_monthly_{title}")
        
    with t_alloc:
        if weights_history:
            # Last Weights Analysis
            last_w = weights_history[-1].to_frame('Weight')
            last_w['Sector'] = last_w.index.map(loader.sector_map).fillna('Other')
            
            # Unified Mapping for Pie Chart
            CRSP_MAPPING = {
                "Consumer Staples": "Basic Needs", "Consumer Defensive": "Basic Needs",
                "Consumer Discretionary": "Lifestyle & Luxury", "Consumer Cyclical": "Lifestyle & Luxury",
                "Financials": "Financial Services", "Information Technology": "Technology",
                "Materials": "Basic Materials", "Health Care": "Healthcare"
            }
            last_w['Unified_Sector'] = last_w['Sector'].map(lambda x: CRSP_MAPPING.get(x, x))
            
            col_a1, col_a2 = st.columns(2)
            
            with col_a1:
                st.subheader("Current Sector Exposure")
                sector_alloc = last_w.groupby('Unified_Sector')['Weight'].sum().reset_index()
                fig_pie = px.pie(sector_alloc, values='Weight', names='Unified_Sector', hole=0.4, color_discrete_sequence=px.colors.qualitative.Prism)
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(showlegend=False, template="plotly_dark")
                st.plotly_chart(fig_pie, use_container_width=True, key=f"chart_sector_{title}")
                
            with col_a2:
                st.subheader("Top 10 Holdings")
                top_10 = last_w.sort_values('Weight', ascending=False).head(10)
                # Add Name if available
                if loader.name_map:
                    top_10['Name'] = top_10.index.map(lambda x: loader.name_map.get(x, x))
                else:
                    top_10['Name'] = top_10.index
                    
                fig_bar = px.bar(top_10, x='Weight', y='Name', orientation='h', text_auto='.1%')
                fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0))
                fig_bar.update_traces(marker_color='#22D3EE')
                st.plotly_chart(fig_bar, use_container_width=True, key=f"chart_top10_{title}")
            
            st.subheader("Allocation History (Wave Graph)")
            plot_sector_allocation(weights_history, dates, loader.sector_map, "Sector Allocation Over Time", key=f"chart_alloc_{title}")
            
            # Compare Wave Graphs if Resampling
            if benchmark_results:
                st.markdown("---")
                st.subheader(f"Comparison: {bench_name} Allocation")
                plot_sector_allocation(benchmark_results['weights'], benchmark_results['dates'], loader.sector_map, f"{bench_name} Allocation", key=f"chart_alloc_base_{title}")
            
            st.markdown("---")
            st.subheader("Full Asset List")
            
            full_list = last_w.copy()
            full_list['Ticker'] = full_list.index
            
            # Add Name (Exact logic from Utility)
            full_list['Name'] = full_list.index.map(lambda x: loader.name_map.get(x, x) if loader.name_map else x)
            
            full_list = full_list[['Ticker', 'Name', 'Unified_Sector', 'Weight']]
            full_list = full_list.sort_values('Weight', ascending=False)
            
            # Reset Index to start from 1
            full_list.index = range(1, len(full_list) + 1)
            
            st.dataframe(
                full_list.style.format({'Weight': '{:.2%}'}),
                use_container_width=True
            )
            
    with t_tech:
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
            
    with t_det:
        # Advanced Metrics Table (Aligned with Expert Risk)
        st.markdown("### 📊 Advanced Performance Metrics")
        
        var_95 = np.percentile(rets, 5)
        cvar_95 = rets[rets <= var_95].mean()
        skew = rets.skew()
        kurt = rets.kurtosis()
        
        sortino = (ann_ret - 0.02) / (rets[rets < 0].std() * np.sqrt(12)) if rets[rets < 0].std() > 0 else 0
        calmar = ann_ret / abs(max_drawdown) if max_drawdown != 0 else 0
        
        win_rate = (rets > 0).mean()
        best_month = rets.max()
        worst_month = rets.min()
        
        # Configuration Table
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            st.markdown("### 🛠️ Configuration")
            cfg = loader.config
            
            # Helper to safely get param
            def get_param(key, default):
                if 'last_frontier_params' in st.session_state:
                    return st.session_state.last_frontier_params.get('optimizer_params', {}).get(key, default)
                return default

            mdd_date = drawdown_series.idxmin().date()
            
            config_data = {
                "Parameter": [
                    "Data Source", "Universe Size", "Date Range", "Lookback Period",
                    "Selection Method", "Max Turnover", "Avg. Realized Turnover", "Transaction Cost",
                    "Covariance Method", "Solver", "Max Drawdown Date"
                ],
                "Value": [
                    cfg.get('source', 'N/A'),
                    f"{cfg.get('n_stocks', 'N/A')} Assets",
                    f"{cfg.get('test_start')} to {cfg.get('test_end')}",
                    f"{cfg.get('lookback_months', 24)} Months",
                    cfg.get('selection_method', 'N/A').replace('_', ' ').title(),
                    f"{get_param('max_turnover', 'N/A')}",
                    f"{turnover:.2%}",
                    f"{st.session_state.last_frontier_params.get('cost_bps', 0)*10000:.0f} bps" if 'last_frontier_params' in st.session_state else "N/A",
                    f"{get_param('cov_method', 'ledoit')}",
                    f"{get_param('solver', 'N/A')}",
                    f"{mdd_date}"
                ]
            }
            st.table(pd.DataFrame(config_data).set_index("Parameter"))

        with col_d2:
            st.markdown("### 📈 Statistics")
            cum_ret = (1 + rets).prod() - 1
            
            perf_data = {
                "Metric": [
                    "Cumulative Return", "Annualized Return", "Annualized Volatility",
                    "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
                    "Max Drawdown", "Value at Risk (95%)", "CVaR (95%)",
                    "Skewness", "Kurtosis", "Win Rate (Months)",
                    "Best Month", "Worst Month"
                ],
                "Value": [
                    f"{values.iloc[-1]:.2f}x", f"{ann_ret:.2%}", f"{ann_vol:.2%}",
                    f"{sharpe:.2f}", f"{sortino:.2f}", f"{calmar:.2f}",
                    f"{max_drawdown:.2%}", f"{var_95:.2%}", f"{cvar_95:.2%}",
                    f"{skew:.2f}", f"{kurt:.2f}", f"{win_rate:.1%}",
                    f"{best_month:.2%}", f"{worst_month:.2%}"
                ]
            }
            st.table(pd.DataFrame(perf_data).set_index("Metric"))
            
    with t_exp:
        st.subheader("📥 Export Results")
        csv_values = values.to_csv()
        st.download_button(
            label="Download Portfolio Values (CSV)",
            data=csv_values,
            file_name="portfolio_values.csv",
            mime="text/csv",
            key=f"dl_csv_val_{title}"
        )

# --- SIDEBAR CONFIGURATION ---
st.sidebar.page_link("pages/experts/landing_expert.py", label="Back to Expert Home", icon="⬅️")
st.sidebar.header("1. Data Configuration")

# Defaults
default_source = 0
default_start = pd.to_datetime("2010-01-01")
default_end = pd.to_datetime("2024-12-31")
default_n_assets = 50
default_method = 0 # Top Market Cap

# Restore defaults from session state if available
if 'loader' not in st.session_state:
    st.session_state.loader = None

if st.session_state.loader is not None:
    cfg = st.session_state.loader.config
    if cfg.get('source') == 'CRSP': default_source = 1
    default_start = pd.to_datetime(cfg['test_start'])
    default_end = pd.to_datetime(cfg['test_end'])
    default_n_assets = int(cfg['n_stocks'])
    if cfg.get('selection_method') == 'random': default_method = 1

    if cfg.get('selection_method') == 'random': default_method = 1

# Data Source (Static)
st.sidebar.caption("Data Source: YFinance")
data_source = "YFINANCE"

col1, col2 = st.sidebar.columns(2)
test_start = col1.date_input("Start Date", default_start, min_value=pd.to_datetime("2010-01-01"), max_value=pd.to_datetime("2024-12-31"))
test_end = col2.date_input("End Date", default_end, min_value=pd.to_datetime("2010-01-01"), max_value=pd.to_datetime("2024-12-31"))

st.sidebar.subheader("Asset Selection")
selection_method = st.sidebar.radio("Method", ["Top Market Cap", "Random Selection"], index=default_method)
n_stocks = st.sidebar.slider("Number of Assets", min_value=10, max_value=100, value=default_n_assets, step=10)
# lookback_months removed as per request (default 24 used internally)

# Construct Config from Sidebar Inputs
config = {
    'source': str(data_source),
    'data_dir': 'data/yfinance', # Standardized path
    'input_file': 'financial_universe_clean.parquet',
    'test_start': str(test_start),
    'test_end': str(test_end),
    'lookback_months': 24, # Default internal value
    'selection_method': 'top_market_cap' if selection_method == "Top Market Cap" else 'random',
    'n_stocks': int(n_stocks),
    'rf_file': 'risk_free.parquet'
}

# Auto-Load / Reactive Reload Logic
should_reload = False
if st.session_state.loader is None:
    should_reload = True
else:
    # Check if critical parameters changed
    prev_config = st.session_state.loader.config
    
    # Robust comparison
    if (str(prev_config.get('source')) != config['source'] or
        str(prev_config.get('test_start')) != config['test_start'] or
        str(prev_config.get('test_end')) != config['test_end'] or
        int(prev_config.get('n_stocks')) != config['n_stocks'] or
        str(prev_config.get('selection_method')) != config['selection_method']):
        should_reload = True

if should_reload:
    with st.spinner("Updating Universe Data..."):
        try:
            st.session_state.loader = load_data(config)
            st.rerun()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

if st.session_state.loader:
    st.sidebar.success(f"✅ Loaded: {len(st.session_state.loader.returns_matrix.columns)} Assets")

st.sidebar.markdown("---")
st.sidebar.header("2. Objective")
objective = st.sidebar.radio("Optimization Goal", ["Minimize Variance", "Maximize Sharpe Ratio"])

st.sidebar.markdown("---")
st.sidebar.markdown("---")
st.sidebar.header("3. Constraints")

# Sector Constraints
enable_sector_constraints = st.sidebar.checkbox("Enable Sector Constraints")
sector_constraints = None

if enable_sector_constraints:
    with st.sidebar.expander("Sector Allocation Limits", expanded=True):
        UNIFIED_SECTORS = [
            "Basic Needs", "Lifestyle & Luxury", "Financial Services", "Technology",
            "Healthcare", "Energy", "Industrials", "Basic Materials",
            "Communication Services", "Real Estate", "Utilities", "Fixed Income"
        ]
        CRSP_MAPPING = {
            "Consumer Staples": "Basic Needs", "Consumer Defensive": "Basic Needs",
            "Consumer Discretionary": "Lifestyle & Luxury", "Consumer Cyclical": "Lifestyle & Luxury",
            "Financials": "Financial Services", "Information Technology": "Technology",
            "Materials": "Basic Materials", "Health Care": "Healthcare"
        }
        
        loader = st.session_state.loader
        if loader and loader.sector_map:
            sector_constraints = {}
            # Remap sector map in loader temporarily for constraints
            # (In a real app, we might want to do this more cleanly)
            
            for sector in sorted(UNIFIED_SECTORS):
                min_w, max_w = st.slider(f"{sector}", 0.0, 1.0, (0.0, 1.0), key=f"sec_{sector}")
                if (min_w, max_w) != (0.0, 1.0):
                    sector_constraints[sector] = (min_w, max_w)
        else:
            st.sidebar.warning("No sector data available.")

# Turnover Constraint
max_turnover = st.sidebar.slider("Max Turnover (Monthly)", 0.05, 1.0, 0.20, 0.05, help="Limit the percentage of portfolio value traded each month.")

# Transaction Cost
cost_bps = st.sidebar.number_input("Transaction Cost (bps)", 0, 100, 10, help="Basis points per trade (e.g., 10 bps = 0.10%)") / 10000

# Risk Free Rate
use_dynamic_rf = st.sidebar.checkbox("Use Dynamic Risk-Free Rate (FRED)", value=True, help="If checked, uses Fama-French RF. Else uses 0%.")

st.sidebar.markdown("---")
st.sidebar.markdown("---")
# Advanced Method
with st.sidebar.expander("Advanced Method"):
    # Covariance Method
    cov_method_display = st.selectbox("Covariance Method", ["Ledoit-Wolf", "Historic"], index=0)
    cov_method_map = {"Ledoit-Wolf": "ledoit", "Historic": "hist"}
    cov_method = cov_method_map[cov_method_display]

# Solver Selection
solver = "CLARABEL"

st.sidebar.markdown("---")
st.sidebar.header("4. Execution")
run_btn = st.sidebar.button("▶️ Run Optimization", type="primary", use_container_width=True)

# --- MAIN LOGIC ---
# Tabs
tab_opt, tab_resamp = st.tabs(["📊 Optimization Results", "✨ Resampling"])

if run_btn:
    if st.session_state.loader:
        with st.spinner("Optimizing..."):
            # Select Optimizer
            if objective == "Minimize Variance":
                opt_class = MinVarianceOptimizer
                opt_params = {'solver': solver, 'cov_method': cov_method, 'max_turnover': max_turnover}
            else: # Maximize Sharpe Ratio
                opt_class = MaxSharpeOptimizer
                opt_params = {'solver': solver, 'cov_method': cov_method, 'max_turnover': max_turnover}
            
            # RF Handling
            rf_series = st.session_state.loader.get_rf_data() if use_dynamic_rf else None
            optimize_kwargs = {'rf': 0.0} # Default
            
            # Run
            results = run_backtest(
                st.session_state.loader,
                opt_class,
                opt_params,
                sector_constraints,
                cost_bps=cost_bps,
                use_dynamic_rf=use_dynamic_rf, # Pass this arg
                optimize_kwargs=optimize_kwargs
            )
            
            # Attach RF Series to results for metrics
            if rf_series is not None:
                results['rf_series'] = rf_series
            
            st.session_state.frontier_results = results
            st.session_state.last_frontier_params = {
                'optimizer_class': opt_class,
                'optimizer_params': opt_params,
                'sector_constraints': sector_constraints,
                'cost_bps': cost_bps,
                'optimize_kwargs': optimize_kwargs
            }
            st.toast("Optimization Complete!")

with tab_opt:
    if 'frontier_results' in st.session_state:
        display_frontier_results(st.session_state.frontier_results, st.session_state.loader, "Optimization Results")
    else:
        st.info("Run optimization to see results.")

with tab_resamp:
    st.markdown("### Michaud Resampling")
    n_sims = st.slider("Number of Simulations", 1, 10, 1, key="n_sims_frontier")
    
    if st.button("▶️ Run Resampling", key="btn_resamp_frontier"):
        if 'last_frontier_params' in st.session_state:
            with st.spinner(f"Running {n_sims} simulations..."):
                params = st.session_state.last_frontier_params
                res_results = run_backtest(
                    st.session_state.loader,
                    params['optimizer_class'],
                    params['optimizer_params'],
                    params['sector_constraints'],
                    cost_bps=0.0010,
                    use_dynamic_rf=True, # Default for resampling
                    enable_resampling=True,
                    n_simulations=n_sims
                )
                st.session_state.frontier_resamp_results = res_results
                
    if 'frontier_resamp_results' in st.session_state:
        base = st.session_state.get('frontier_results')
        display_frontier_results(st.session_state.frontier_resamp_results, st.session_state.loader, "Resampled Results", benchmark_results=base)


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from src.products.definitions import run_strategy

# CSS for Institutional Design
st.markdown("""
<style>
    [data-testid="stSidebar"] {display: none;}
    
    /* Global Font & Colors */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #f8fafc; /* Slate-50 */
    }
    p, li {
        color: #cbd5e1; /* Slate-300 */
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    /* Marketing Card - Glassmorphism Dark */
    .marketing-card {
        background: rgba(30, 41, 59, 0.7); /* Slate-800 with opacity */
        border: 1px solid #334155; /* Slate-700 */
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        backdrop-filter: blur(10px);
    }
    .marketing-card h4 {
        color: #f97316; /* Orange-500 */
        margin-bottom: 16px;
        font-weight: 600;
    }
    .feature-item {
        display: flex;
        align-items: center;
        margin-bottom: 12px;
        color: #e2e8f0; /* Slate-200 */
    }
    .feature-icon {
        margin-right: 12px;
        font-size: 1.2rem;
        color: #f97316; /* Orange-500 */
    }
    
    /* Metric Cards */
    .metric-container {
        background-color: #0f172a; /* Slate-900 */
        border: 1px solid #1e293b; /* Slate-800 */
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-label {
        color: #94a3b8; /* Slate-400 */
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #f1f5f9; /* Slate-100 */
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* Button Styling */
    .stButton button {
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.2s;
    }
    
    /* Simulation Button (Primary) - Royal Blue */
    .stButton button[kind="primary"] {
        background: linear-gradient(90deg, #1e3a8a 0%, #172554 100%) !important;
        border: 1px solid #3b82f6 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: 0.05em !important;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.4) !important;
    }
    .stButton button[kind="primary"]:hover {
        background: linear-gradient(90deg, #1e40af 0%, #1e3a8a 100%) !important;
        border-color: #60a5fa !important;
        box-shadow: 0 6px 16px rgba(30, 58, 138, 0.6) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Simulator Card */
    .simulator-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Back Button
if st.button("⬅️ Back to Home"):
    st.switch_page("pages/landing.py")

# --- Marketing Section ---
st.title("⚡ Global Max Return")
st.markdown("<h3 style='color: #94a3b8; font-weight: 400; margin-top: -10px;'>Unconstrained Capital Growth</h3>", unsafe_allow_html=True)

st.markdown("---")

col_m1, col_m2 = st.columns([1.8, 1])

with col_m1:
    st.markdown("""
    <div style="padding-right: 20px;">
        <p>
            The <b>Global Max Return</b> strategy is the ultimate expression of aggressive investing. 
            It seeks to maximize absolute returns with minimal regard for volatility, targeting the highest-performing assets in the universe.
        </p>
        <p>
            By utilizing a near-zero risk aversion parameter, this portfolio is designed to capture the full upside of market movements, making it suitable only for those with the highest risk tolerance.
        </p>
        <ul style="list-style-type: none; padding-left: 0;">
            <li style="margin-bottom: 10px;">✅ <b>Absolute Return Focus</b>: Prioritizes growth above all else.</li>
            <li style="margin-bottom: 10px;">✅ <b>High Conviction</b>: Concentrated bets on top-performing sectors.</li>
            <li>✅ <b>Long-Term Wealth</b>: Designed to compound capital at the highest possible rate.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("💡 **Investment Profile**: Ultra-aggressive investors seeking maximum wealth accumulation over long horizons.")

with col_m2:
    st.markdown("""
    <div class="marketing-card">
        <h4>Key Characteristics</h4>
        <div class="feature-item"><span class="feature-icon">⚡</span> Max Aggression</div>
        <div class="feature-item"><span class="feature-icon">📉</span> Near-Zero Risk Aversion</div>
        <div class="feature-item"><span class="feature-icon">🔄</span> Monthly Rebalancing</div>
        <div class="feature-item"><span class="feature-icon">🚀</span> Uncapped Potential</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- Action Section ---
if 'show_results_aggressive_3' not in st.session_state:
    st.session_state.show_results_aggressive_3 = False

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    if st.button("🚀 Simulate Historical Performance (2010-2024)", use_container_width=True, type="primary"):
        st.session_state.show_results_aggressive_3 = True

if st.session_state.show_results_aggressive_3:
    st.markdown("---")
    
    # Run Strategy (Cached in Session State)
    if 'strategy_results_aggressive_3' not in st.session_state:
        with st.spinner("🔄 Optimizing Portfolio & Running Backtest..."):
            st.session_state.strategy_results_aggressive_3 = run_strategy("Aggressive MaxRet")
    
    results = st.session_state.strategy_results_aggressive_3

    values = results['values']
    weights_history = results['weights_history']
    loader = results['loader']

    # --- Metrics Calculation ---
    def calculate_drawdown(vals):
        peak = vals.cummax()
        return (vals - peak) / peak

    def calc_metrics(vals):
        rets = vals.pct_change().dropna()
        ann_r = rets.mean() * 12
        ann_v = rets.std() * np.sqrt(12)
        shp = (ann_r - 0.02) / ann_v if ann_v > 0 else 0
        dd = calculate_drawdown(vals).min()
        
        # Additional Metrics
        best_month = rets.max()
        worst_month = rets.min()
        win_rate = (rets > 0).mean()
        
        return vals.iloc[-1], ann_r, ann_v, shp, dd, best_month, worst_month, win_rate

    cum_ret, ann_ret, ann_vol, sharpe, max_drawdown, best_m, worst_m, win_rate = calc_metrics(values)

    # --- Metric Cards ---
    m1, m2, m3, m4 = st.columns(4)

    def metric_card(label, value):
        return f"""
        <div class="metric-container">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """

    with m1: st.markdown(metric_card("Ann. Return", f"{ann_ret:.2%}"), unsafe_allow_html=True)
    with m2: st.markdown(metric_card("Ann. Volatility", f"{ann_vol:.2%}"), unsafe_allow_html=True)
    with m3: st.markdown(metric_card("Sharpe Ratio", f"{sharpe:.2f}"), unsafe_allow_html=True)
    with m4: st.markdown(metric_card("Max Drawdown", f"{max_drawdown:.2%}"), unsafe_allow_html=True)

    st.markdown("###")

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance Analysis", "🥧 Portfolio Allocation", "📋 Technical Details", "💰 Simulator"])

    with tab1:
        st.subheader("Cumulative Growth (Base 100)")
        
        cum_ret_series = values * 100
        drawdown_series = calculate_drawdown(values)
        
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.7, 0.3],
                            subplot_titles=("Portfolio Value", "Drawdown"))
        
        # Institutional Orange Theme for Aggressive
        fig.add_trace(go.Scatter(x=cum_ret_series.index, y=cum_ret_series, name='Max Return Strategy', 
                                 line=dict(color='#f97316', width=2)), row=1, col=1) # Orange-500
        
        fig.add_trace(go.Scatter(x=drawdown_series.index, y=drawdown_series, name='Drawdown',
                                 line=dict(color='#ef4444', width=1), fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.1)'), row=2, col=1)
        
        fig.update_layout(height=600, template="plotly_dark", hovermode="x unified", 
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor='#334155')
        
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if weights_history:
            final_weights = weights_history[-1]
            w_df = final_weights.to_frame(name='Weight')
            
            # Map Names & Sectors
            w_df['Company'] = w_df.index.map(lambda x: loader.name_map.get(x, x) if loader.name_map else x)
            w_df['Sector'] = w_df.index.map(loader.sector_map).fillna('Other')
            
            col_a1, col_a2 = st.columns([1, 1])
            
            with col_a1:
                st.markdown("#### Sector Exposure")
                sector_alloc = w_df.groupby('Sector')['Weight'].sum().reset_index()
                fig_pie = px.pie(sector_alloc, values='Weight', names='Sector', hole=0.5, 
                                 color_discrete_sequence=px.colors.qualitative.Prism)
                fig_pie.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col_a2:
                st.markdown("#### Top Holdings")
                top_holdings = w_df[w_df['Weight'] > 0.001].sort_values('Weight', ascending=False).head(10)
                display_df = top_holdings[['Company', 'Sector', 'Weight']].copy()
                st.dataframe(
                    display_df.style.format({'Weight': "{:.2%}"}),
                    use_container_width=True,
                    hide_index=True
                )

    with tab3:
        from src.products.ui_utils import render_performance_metrics
        render_performance_metrics(values)
            
    with tab4:
        st.markdown("#### 💰 Investment Simulator")
        
        st.markdown("""
        <div class="simulator-card">
            <p style="color: #94a3b8; margin-bottom: 10px;">Project your potential returns based on historical performance.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_sim1, col_sim2 = st.columns([1, 2])
        
        with col_sim1:
            initial_inv = st.number_input("Initial Investment ($)", min_value=1000, value=10000, step=1000)
            horizon_years = st.slider("Investment Horizon (Years)", 1, 10, 5)
            
            # Historical Calculation
            hist_multiple = values.iloc[-1]
            hist_final = initial_inv * hist_multiple
            hist_profit = hist_final - initial_inv
            
            st.markdown(f"""
            <div style="margin-top: 20px; padding: 15px; background: rgba(249, 115, 22, 0.1); border-radius: 8px; border: 1px solid #f97316;">
                <h5 style="color: #f97316; margin:0;">Historical Result (2010-2024)</h5>
                <h2 style="color: white; margin: 5px 0;">${hist_final:,.0f}</h2>
                <p style="color: #f97316; margin:0;">+{hist_profit:,.0f} (+{hist_multiple*100-100:.0f}%)</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_sim2:
            # Monte Carlo Simulation
            st.markdown(f"<h3 style='text-align: center; color: #f8fafc; margin-bottom: 20px;'>Future Projection ({horizon_years} Years)</h3>", unsafe_allow_html=True)
            
            # Parameters
            mu = ann_ret
            sigma = ann_vol
            n_sims = 100
            n_steps = horizon_years * 12 # Monthly steps
            dt = 1/12
            
            # Simulation
            paths = []
            np.random.seed(42)
            for _ in range(n_sims):
                price_path = [initial_inv]
                for _ in range(n_steps):
                    # Geometric Brownian Motion
                    drift = (mu - 0.5 * sigma**2) * dt
                    shock = sigma * np.sqrt(dt) * np.random.normal()
                    price = price_path[-1] * np.exp(drift + shock)
                    price_path.append(price)
                paths.append(price_path)
                
            # Plotting
            fig_mc = go.Figure()
            
            # Add paths
            time_axis = np.arange(n_steps + 1) / 12
            for path in paths[:50]: # Show first 50 paths
                fig_mc.add_trace(go.Scatter(x=time_axis, y=path, mode='lines', 
                                          line=dict(color='rgba(249, 115, 22, 0.1)', width=1), showlegend=False))
                
            # Add Mean Path
            mean_path = np.mean(paths, axis=0)
            fig_mc.add_trace(go.Scatter(x=time_axis, y=mean_path, mode='lines', name='Expected Scenario',
                                      line=dict(color='#f97316', width=3)))
            
            # Add 95% Confidence Interval
            upper_bound = np.percentile(paths, 95, axis=0)
            lower_bound = np.percentile(paths, 5, axis=0)
            
            fig_mc.add_trace(go.Scatter(x=time_axis, y=upper_bound, mode='lines', name='Upside (95%)',
                                      line=dict(color='#fb923c', width=1, dash='dash'))) # Orange-400
            fig_mc.add_trace(go.Scatter(x=time_axis, y=lower_bound, mode='lines', name='Downside (5%)',
                                      line=dict(color='#ef4444', width=1, dash='dash')))
            
            fig_mc.update_layout(height=350, template="plotly_dark", 
                               paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               xaxis_title="Years", yaxis_title="Portfolio Value ($)",
                               margin=dict(l=20, r=20, t=20, b=20))
            
            st.plotly_chart(fig_mc, use_container_width=True)
            
            # Final Stats
            avg_final = np.mean([p[-1] for p in paths])
            
            # Styled Expected Value
            st.markdown(f"""
            <div style="text-align: center; margin-top: 20px; padding: 20px; border-top: 1px solid #334155;">
                <p style="color: #94a3b8; font-size: 1.1rem; margin-bottom: 5px;">Expected Value in {horizon_years} years</p>
                <h1 style="color: #f97316; font-size: 3rem; font-weight: 800; margin: 0;">${avg_final:,.0f}</h1>
                <p style="color: #64748b; font-size: 0.9rem; margin-top: 5px;">(Monte Carlo Simulation Mean)</p>
            </div>
            """, unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import riskfolio as rp

# Add src to path
sys.path.append(os.path.abspath('src'))

from portfolio_engine.data_loader import PortfolioDataLoader
from portfolio_engine.optimizers import MinVarianceOptimizer, MinCVaROptimizer, RiskParityOptimizer, ResampledOptimizer
from portfolio_engine.utils import get_optimal_solver
from portfolio_engine.technicals import get_technical_indicators
from portfolio_engine.weights_management import OptimizedWeightUpdater

# --- Custom CSS ---
st.markdown("""
<style>
    [data-testid="stSidebar"] {display: block !important;}

    .metric-card {
        background-color: #1F2937;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #374151;
        margin-bottom: 1rem;
    }
    h5 {color: #9CA3AF; margin-bottom: 0.5rem; font-size: 0.9rem;}
    h2 {color: #F3F4F6; margin: 0; font-size: 1.8rem;}
    
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

# --- Helper Functions ---
def load_data(config):
    """Loads data using PortfolioDataLoader and caches it in session state."""
    loader = PortfolioDataLoader(config)
    loader.load_data()
    return loader

def calculate_drawdown(series):
    """Calculates drawdown series."""
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    return drawdown

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

def run_backtest(loader, optimizer_class, optimizer_params, sector_constraints, cost_bps, use_dynamic_rf, optimize_kwargs={}, benchmark_series=None, enable_resampling=False, n_simulations=100):
    """
    Runs a walk-forward backtest (monthly rebalancing).
    """
    # 1. Prepare Data
    returns_df = loader.returns_matrix
    sector_map = loader.sector_map
    
    # Align dates
    test_start_dt = pd.to_datetime(loader.config['test_start'])
    test_end_dt = pd.to_datetime(loader.config['test_end'])
    
    # Filter for simulation period
    sim_returns = returns_df.loc[test_start_dt:test_end_dt]
    simulation_dates = sim_returns.index
    
    # Initialize Optimizer
    optimizer = optimizer_class(**optimizer_params)
    
    # Wrap with Resampling if enabled
    if enable_resampling:
        optimizer = ResampledOptimizer(optimizer, n_simulations=n_simulations)
    
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
        status_text.text(f"Optimizing: {date.date()}")
        
        # Define Lookback Window
        window_start = date - pd.DateOffset(months=loader.config['lookback_months'])
        hist_returns = returns_df.loc[window_start:date].iloc[:-1] # Exclude current month
        
        if len(hist_returns) < 12:
            portfolio_values.append(portfolio_values[-1])
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
            target_weights = optimizer.optimize(
                returns=opt_returns,
                prev_weights=current_weights, # Pass prev_weights for turnover constraint
                sector_constraints=sector_constraints,
                sector_map=sector_map,
                **optimize_kwargs
            )
        except Exception as e:
            # Fallback to current weights or Equal Weight
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
    
    # Correctly construct RF Series aligned with simulation dates
    rf_aligned_series = None
    if use_dynamic_rf and loader.rf_series is not None:
        # Reindex to match simulation dates (filling missing with 0)
        # This ensures we get a Series, not an Index, and handles alignment/duplicates naturally
        rf_aligned_series = loader.rf_series.reindex(simulation_dates).fillna(0.0)

    results = {
        'values': pd.Series(portfolio_values, index=[simulation_dates[0] - pd.DateOffset(days=1)] + list(simulation_dates)),
        'turnover': np.mean(turnover_history) if turnover_history else 0.0,
        'weights': weights_history,
        'dates': simulation_dates,
        'rf_series': rf_aligned_series
    }
    
    return results

# ==============================================================================
# SIDEBAR CONFIGURATION
# ==============================================================================
st.sidebar.page_link("pages/experts/landing_expert.py", label="Back to Expert Home", icon="⬅️")
st.sidebar.header("1. Data Configuration")

# Defaults
default_source = 0
default_start = pd.to_datetime("2010-01-01")
default_end = pd.to_datetime("2024-12-31")
default_n_assets = 50
default_method = 0 # Top Market Cap

# Restore defaults from session state if available
if 'loader' in st.session_state and st.session_state.loader is not None:
    cfg = st.session_state.loader.config
    if cfg.get('source') == 'CRSP': default_source = 1
    default_start = pd.to_datetime(cfg['test_start'])
    default_end = pd.to_datetime(cfg['test_end'])
    default_n_assets = int(cfg['n_stocks'])
    if cfg.get('selection_method') == 'random': default_method = 1

    if cfg.get('selection_method') == 'random': default_method = 1

# Dynamic Source Availability
available_sources = ["YFINANCE"]
crsp_paths = ["data/processed/returns_liquid_nyse80.parquet", "../data/processed/returns_liquid_nyse80.parquet"]
crsp_available = any(os.path.exists(p) for p in crsp_paths)

if crsp_available:
    available_sources.append("CRSP")

if default_source == 1 and not crsp_available:
    default_source = 0

data_source = st.sidebar.selectbox("Data Source", available_sources, index=default_source)

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start Date", default_start)
end_date = col2.date_input("End Date", default_end)

if st.sidebar.button("🔄 Reload Data", use_container_width=True):
    st.session_state.loader = None
    st.rerun()

st.sidebar.subheader("Asset Selection")
selection_method = st.sidebar.radio("Method", ["Top Market Cap", "Random Selection"], index=default_method)
n_assets = st.sidebar.slider("Number of Assets", min_value=10, max_value=500, value=default_n_assets, step=10)

# --- Constraints ---
st.sidebar.header("2. Constraints")
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

        if 'loader' in st.session_state and st.session_state.loader is not None:
            loader = st.session_state.loader
            if loader.sector_map:
                sector_constraints = {}
                remapped_map = {}
                for ticker, sec in loader.sector_map.items():
                    remapped_map[ticker] = CRSP_MAPPING.get(sec, sec)
                loader.sector_map = remapped_map
                
                for sector in sorted(UNIFIED_SECTORS):
                    min_w, max_w = st.slider(f"{sector}", 0.0, 1.0, (0.0, 1.0), key=f"sec_{sector}")
                    sector_constraints[sector] = (min_w, max_w)
            else:
                st.sidebar.warning("No sector data available. Load data first.")
        else:
            st.sidebar.info("Load data to see sectors.")

# --- Strategy Parameters (Centralized Model Selection) ---
st.sidebar.header("3. Strategy Parameters")

# Model Selection
risk_model_type = st.sidebar.selectbox("Risk Model", ["Absolute Risk", "Relative Risk", "Risk Budgeting"])

# Objective Selection based on Model
optimizer_mode = None
confidence_level = 0.95 # Default
risk_budget = None # Default

if risk_model_type == "Absolute Risk":
    optimizer_mode = st.sidebar.radio("Objective", ["Minimum Variance", "Minimum CVaR"])
    if optimizer_mode == "Minimum CVaR":
        confidence_level = st.sidebar.slider("Confidence Level (1-alpha)", 0.90, 0.99, 0.95, 0.01)
        st.sidebar.caption(f"Optimizing for the worst {(1-confidence_level):.0%} scenarios.")

elif risk_model_type == "Relative Risk":
    optimizer_mode = "Tracking Error"
    st.sidebar.info("Objective: Minimize Tracking Error vs Benchmark")
    
    # Benchmark Selection
    st.sidebar.markdown("### Benchmark Parameters")
    benchmark_type = st.sidebar.radio("Benchmark", ["Equal Weighted (EW)", "Value Weighted (VW)"], index=0)

elif risk_model_type == "Risk Budgeting":
    optimizer_mode = "Risk Parity"
    st.sidebar.info("Objective: Equal Risk Contribution (ERC)")
    st.sidebar.caption("Allocates risk equally across all assets. This is the most robust approach for Risk Parity.")
    
    # Custom Budgets disabled for now to avoid conflict with Sector Constraints
    # and to simplify the workflow.
    risk_budget = None


# Common Parameters
max_turnover = st.sidebar.slider("Max Monthly Turnover", 0.0, 1.0, 0.20)
long_only = st.sidebar.checkbox("Long Only", value=True)
cost_bps = st.sidebar.number_input("Transaction Cost (bps)", value=10) / 10000
use_dynamic_rf = st.sidebar.checkbox("Use Dynamic Risk-Free Rate", value=False)

# --- Advanced Parameters ---
with st.sidebar.expander("Advanced Parameters"):
    cov_method = st.sidebar.selectbox("Covariance Method", ["ledoit", "hist", "oas"], index=0)

    available_solver = get_optimal_solver()
    solver = st.sidebar.selectbox("Solver", [available_solver, "CLARABEL", "ECOS", "OSQP", "SCS"], index=0)

    if available_solver == 'MOSEK':
        st.sidebar.caption("🚀 **MOSEK Detected & Active**")
    else:
        st.sidebar.caption("🛡️ **CLARABEL Active**")

st.sidebar.markdown("---")
run_btn = st.sidebar.button("▶️ Run Optimization", type="primary", use_container_width=True)

# ==============================================================================
# MAIN AREA
# ==============================================================================
st.title("🛡️ Risk Architecture")

# Initialize Session State
if 'loader' not in st.session_state:
    st.session_state.loader = None

# Construct Config
config = {
    'source': str(data_source),
    'data_dir': '../data/yfinance' if data_source == 'YFINANCE' else '../data/processed',
    'input_file': 'financial_universe_clean.parquet' if data_source == 'YFINANCE' else 'returns_liquid_nyse80.parquet',
    'test_start': str(start_date),
    'test_end': str(end_date),
    'lookback_months': 24,
    'selection_method': 'top_market_cap' if selection_method == "Top Market Cap" else 'random',
    'n_stocks': int(n_assets),
    'rf_file': '../data/processed/risk_free_daily_2000_2025.parquet'
}

# Auto-Load Logic
should_reload = False
if st.session_state.loader is None:
    should_reload = True
else:
    prev_config = st.session_state.loader.config
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
            
            # Clear previous optimization results to avoid stale metrics
            keys_to_clear = ['risk_results', 'risk_optimizer_mode']
            for k in keys_to_clear:
                if k in st.session_state:
                    del st.session_state[k]
            
            st.rerun()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

# Get Data
loader = st.session_state.loader
returns, caps, sector_map, name_map = loader.get_matrices()

if returns is None or returns.empty:
    st.error("No data available. Please check your configuration.")
    st.stop()

st.sidebar.success(f"Data Loaded: {returns.shape[1]} Assets")



# --- OPTIMIZATION LOGIC ---
if run_btn:
    with st.spinner(f"Running Backtest ({optimizer_mode})..."):
        try:
            optimizer_class = None
            optimizer_params = {}
            optimize_kwargs = {}
            benchmark_series = None
            
            # --- 1. Absolute Risk ---
            if risk_model_type == "Absolute Risk":
                if optimizer_mode == "Minimum Variance":
                    optimizer_class = MinVarianceOptimizer
                    optimizer_params = {
                        'cov_method': cov_method,
                        'max_turnover': max_turnover,
                        'long_only': long_only,
                        'solver': solver
                    }
                elif optimizer_mode == "Minimum CVaR":
                    optimizer_class = MinCVaROptimizer
                    optimizer_params = {
                        'cov_method': cov_method,
                        'max_turnover': max_turnover,
                        'long_only': long_only,
                        'solver': solver,
                        # MinCVaR takes alpha in optimize(), but we can pass it via kwargs if modified,
                        # or we handle it in run_backtest by checking instance type.
                        # Actually MinCVaR init doesn't take alpha, optimize() does.
                        # We need to handle this in run_backtest or wrapper.
                    }
            
            # --- 2. Relative Risk (Placeholder) ---
            # --- 2. Relative Risk (Tracking Error) ---
            elif risk_model_type == "Relative Risk":
                # 1. Calculate Benchmark Series
                benchmark_series = None
                if benchmark_type == "Equal Weighted (EW)":
                    # EW of the universe
                    benchmark_series = returns.mean(axis=1)
                elif benchmark_type == "Value Weighted (VW)":
                    # VW of the universe
                    if caps is not None and not caps.empty:
                        # Align caps to returns
                        caps_aligned = caps.reindex(returns.index).reindex(returns.columns, axis=1).fillna(0)
                        # Calculate VW Return: sum(w * r) where w = cap / total_cap
                        total_cap = caps_aligned.sum(axis=1)
                        # Avoid division by zero
                        total_cap = total_cap.replace(0, np.nan)
                        
                        # Weighted Sum
                        weighted_rets = (returns * caps_aligned).sum(axis=1)
                        benchmark_series = weighted_rets / total_cap
                        benchmark_series = benchmark_series.fillna(0) # Fill gaps
                    else:
                        st.warning("Market Caps not available. Falling back to EW Benchmark.")
                        benchmark_series = returns.mean(axis=1)
                
                # 2. Setup Optimizer (MinVariance of Excess Returns)
                optimizer_class = MinVarianceOptimizer
                optimizer_params = {
                    'cov_method': cov_method,
                    'max_turnover': max_turnover,
                    'long_only': long_only,
                    'solver': solver,
                    'risk_aversion': None # MinRisk (Min TE)
                }
                
            # --- 3. Risk Budgeting (Placeholder) ---
            # --- 3. Risk Budgeting ---
            elif risk_model_type == "Risk Budgeting":
                # UI is handled in Sidebar, we just use the result
                optimizer_class = RiskParityOptimizer
                optimizer_params = {
                    'cov_method': cov_method,
                    'max_turnover': max_turnover,
                    'long_only': long_only,
                    'solver': solver
                }
                
                optimize_kwargs['risk_budget'] = risk_budget

            if optimizer_class:
                # Run Backtest
                # Note: MinCVaR needs 'alpha' passed to optimize.
                # We can subclass or wrap, but for now let's modify run_backtest to handle kwargs?
                # Or just instantiate and set a property if possible.
                # MinCVaROptimizer.optimize takes alpha.
                # Let's make a quick wrapper or handle inside run_backtest loop?
                # Better: Pass kwargs to run_backtest.
                
                # Hack for MinCVaR alpha:
                # We'll modify run_backtest to accept optimize_kwargs
                
                # Re-defining run_backtest locally to support alpha injection would be cleaner
                # but for now let's assume we can pass it.
                
                # Actually, MinCVaROptimizer doesn't store alpha in init.
                # We need to pass it to optimize().
                
                # Let's update run_backtest signature in this file to accept optimize_kwargs
                
                # optimize_kwargs is already initialized
                if optimizer_mode == "Minimum CVaR":
                    optimize_kwargs['alpha'] = 1 - confidence_level

                # Custom run_backtest call
                # We need to modify the run_backtest function defined above to accept optimize_kwargs
                
                # Redefining run_backtest inside the script to support kwargs
                results = run_backtest(
                    loader, 
                    optimizer_class, 
                    optimizer_params, 
                    sector_constraints, 
                    cost_bps, 
                    use_dynamic_rf,
                    optimize_kwargs,
                    benchmark_series, # Pass Benchmark
                    False, # enable_resampling (False for base run)
                    100 # n_simulations (Ignored)
                )
                
                # --- Post-Processing: Calculate Benchmarks for Display ---
                # Always calculate EW and VW (if possible) for comparison
                bench_ew = returns.mean(axis=1)
                bench_vw = None
                if caps is not None and not caps.empty:
                    caps_aligned = caps.reindex(returns.index).reindex(returns.columns, axis=1).fillna(0)
                    total_cap = caps_aligned.sum(axis=1).replace(0, np.nan)
                    bench_vw = (returns * caps_aligned).sum(axis=1) / total_cap
                    bench_vw = bench_vw.fillna(0)
                
                results['bench_ew'] = bench_ew
                results['bench_vw'] = bench_vw
                results['active_benchmark_type'] = benchmark_type if risk_model_type == "Relative Risk" else None

                # Store params for Resampling
                st.session_state.last_run_params = {
                    'optimizer_class': optimizer_class,
                    'optimizer_params': optimizer_params,
                    'optimize_kwargs': optimize_kwargs,
                    'benchmark_series': benchmark_series
                }

                st.session_state.risk_results = results
                st.session_state.risk_optimizer_mode = optimizer_mode

        except Exception as e:
            st.error(f"Optimization Failed: {e}")
            st.exception(e)

# --- DISPLAY LOGIC ---
# --- DISPLAY LOGIC (Refactored) ---
def display_risk_results(results, loader, title="Optimization Results", benchmark_results=None):
    st.subheader(title)
    values = results['values']
    weights_history = results['weights']
    turnover = results['turnover']
    rf_series = results.get('rf_series')
    
    # Calculate Metrics
    rets = values.pct_change().dropna()
    ann_ret = rets.mean() * 12
    ann_vol = rets.std() * np.sqrt(12)
    
    if rf_series is not None:
        rf_aligned = rf_series.reindex(rets.index).fillna(0)
        excess = rets - rf_aligned
        sharpe = (excess.mean() * 12) / ann_vol if ann_vol > 0 else 0
    else:
        sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
        
    max_drawdown = calculate_drawdown(values).min()
    
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
            "Page": "Expert: Risk Architecture",
            "Annual Return": f"{ann_ret:.2%}",
            "Volatility": f"{ann_vol:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Top Holdings": top_holdings_str
        }
        st.session_state['latest_optimization_context'] = context_data
    except Exception as e:
        print(f"Context Injection Error: {e}")
    
    # Benchmark Metrics (Base Strategy)
    b_cum, b_ret, b_vol, b_sharpe, b_dd = None, None, None, None, None
    if benchmark_results:
        b_vals = benchmark_results['values']
        b_rets = b_vals.pct_change().dropna()
        b_ann_ret = b_rets.mean() * 12
        b_ann_vol = b_rets.std() * np.sqrt(12)
        
        b_rf = benchmark_results.get('rf_series')
        if b_rf is not None:
            b_rf_aligned = b_rf.reindex(b_rets.index).fillna(0)
            b_excess = b_rets - b_rf_aligned
            b_sharpe = (b_excess.mean() * 12) / b_ann_vol if b_ann_vol > 0 else 0
        else:
            b_sharpe = (b_ann_ret - 0.02) / b_ann_vol if b_ann_vol > 0 else 0
            
        b_dd = calculate_drawdown(b_vals).min()

    # --- Metric Cards ---
    m1, m2, m3, m4, m5 = st.columns(5)
    
    with m1:
        delta_val = (ann_ret - b_ann_ret) if b_ret is not None else None # b_ret was not assigned above, fixed logic below
        delta_val = (ann_ret - b_ann_ret) if benchmark_results else None
        delta_str = f"vs Base: {delta_val:+.2%}" if delta_val is not None else ""
        display_metric_card("Ann. Return", f"{ann_ret:.2%}", delta_val=delta_val, delta_text=delta_str)

    with m2:
        delta_val = (ann_vol - b_ann_vol) if benchmark_results else None
        delta_str = f"vs Base: {delta_val:+.2%}" if delta_val is not None else ""
        # Lower vol is better, so invert color logic in display_metric_card? 
        # Actually standard logic is Green = Higher. For Vol, Red = Higher usually.
        # But display_metric_card uses Green for Positive.
        # Let's keep standard: Green if Vol increased? No, that's bad.
        # I'll modify display_metric_card call to invert delta_val for color purposes?
        # Or just let user interpret.
        # In Utility page: st.markdown(metric_card("Ann. Volatility", f"{ann_vol:.2%}", -delta_val if delta_val is not None else None, delta_str), unsafe_allow_html=True)
        # So they invert delta_val for color.
        display_metric_card("Ann. Volatility", f"{ann_vol:.2%}", delta_val=-delta_val if delta_val is not None else None, delta_text=delta_str)

    with m3:
        delta_val = (sharpe - b_sharpe) if benchmark_results else None
        delta_str = f"vs Base: {delta_val:+.2f}" if delta_val is not None else ""
        display_metric_card("Sharpe Ratio", f"{sharpe:.2f}", delta_val=delta_val, delta_text=delta_str)

    with m4:
        delta_val = (max_drawdown - b_dd) if benchmark_results else None
        delta_str = f"vs Base: {delta_val:+.2%}" if delta_val is not None else ""
        display_metric_card("Max Drawdown", f"{max_drawdown:.2%}", delta_val=delta_val, delta_text=delta_str)

    with m5:
        display_metric_card("Avg. Turnover", f"{turnover:.2%}")
    
    # Sub-Tabs for Charts & Details
    subtab_perf, subtab_alloc, subtab_tech, subtab_details, subtab_export = st.tabs(["📈 Performance", "🥧 Allocation", "⚡ Technicals", "📋 Details", "📥 Export"])
    
    with subtab_perf:
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
        
        # Trace 1b: Base Strategy (Gray, Solid) - If Resampling
        if benchmark_results:
            b_series = benchmark_results['values'] * 100
            fig.add_trace(go.Scatter(x=b_series.index, y=b_series, name='Base Strategy',
                                     line=dict(color='#9CA3AF', width=2)), row=1, col=1)

        # Trace 1c: Benchmarks (EW/VW)
        bench_ew = results.get('bench_ew')
        bench_vw = results.get('bench_vw')
        
        if bench_ew is not None:
            # Align and Normalize to 100
            b_ew_cum = (1 + bench_ew.reindex(cum_ret_series.index).fillna(0)).cumprod() * 100
            # Rebase to start at 100
            b_ew_cum = b_ew_cum / b_ew_cum.iloc[0] * 100
            fig.add_trace(go.Scatter(x=b_ew_cum.index, y=b_ew_cum, name='Benchmark (EW)',
                                     line=dict(color='#A78BFA', width=2)), row=1, col=1)

        if bench_vw is not None:
            b_vw_cum = (1 + bench_vw.reindex(cum_ret_series.index).fillna(0)).cumprod() * 100
            b_vw_cum = b_vw_cum / b_vw_cum.iloc[0] * 100
            fig.add_trace(go.Scatter(x=b_vw_cum.index, y=b_vw_cum, name='Benchmark (VW)',
                                     line=dict(color='#F59E0B', width=2)), row=1, col=1)
        
        # Trace 2: Drawdown
        fig.add_trace(go.Scatter(x=drawdown_series.index, y=drawdown_series, name='Drawdown',
                                 line=dict(color='#EF4444', width=1), fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.2)'), row=2, col=1)
        
        if benchmark_results:
            b_dd_series = calculate_drawdown(benchmark_results['values'])
            fig.add_trace(go.Scatter(x=b_dd_series.index, y=b_dd_series, name='Base Drawdown',
                                     line=dict(color='#6B7280', width=1)), row=2, col=1)

        fig.update_layout(height=600, showlegend=True, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True, key=f"chart_cum_perf_{title}")
        
        # Monthly Returns
        st.subheader("Monthly Returns")
        monthly_rets = rets.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        colors = ['#10B981' if r > 0 else '#EF4444' for r in monthly_rets]
        
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=monthly_rets.index, y=monthly_rets, marker_color=colors, name='Return'))
        fig_bar.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), template="plotly_dark", showlegend=False)
        fig_bar.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig_bar, use_container_width=True, key=f"chart_monthly_{title}")

    with subtab_alloc:
        col_a1, col_a2 = st.columns([1, 1])
        
        # Use Final Weights for Pie/Bar
        final_weights = weights_history[-1] if weights_history else pd.Series()
        w_df = final_weights.to_frame(name='Weight')
        
        # Map to Raw Sectors
        raw_sectors = w_df.index.map(loader.sector_map).fillna('Other')
        
        # Apply Unified Mapping
        CRSP_MAPPING = {
            "Consumer Staples": "Basic Needs", "Consumer Defensive": "Basic Needs",
            "Consumer Discretionary": "Lifestyle & Luxury", "Consumer Cyclical": "Lifestyle & Luxury",
            "Financials": "Financial Services", "Information Technology": "Technology",
            "Materials": "Basic Materials", "Health Care": "Healthcare"
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
        plot_sector_allocation(weights_history, results['dates'], loader.sector_map, "Sector Allocation Over Time", key=f"chart_alloc_{title}")

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

    with subtab_tech:
        st.subheader("Technical Indicators")
        
        tech_df = get_technical_indicators(values)
        
        # RSI
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=tech_df.index, y=tech_df['RSI'], name='RSI', line=dict(color='#A78BFA')))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(title="RSI (14)", height=300, template="plotly_dark")
        st.plotly_chart(fig_rsi, use_container_width=True, key=f"chart_rsi_{title}")
        
        # MACD
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=tech_df.index, y=tech_df['MACD'], name='MACD', line=dict(color='#3B82F6')))
        fig_macd.add_trace(go.Scatter(x=tech_df.index, y=tech_df['Signal'], name='Signal', line=dict(color='#F59E0B')))
        fig_macd.update_layout(title="MACD", height=300, template="plotly_dark")
        st.plotly_chart(fig_macd, use_container_width=True, key=f"chart_macd_{title}")
        
        # Stochastic Oscillator
        fig_stoch = go.Figure()
        fig_stoch.add_trace(go.Scatter(x=tech_df.index, y=tech_df['%K'], name='%K', line=dict(color='#10B981')))
        fig_stoch.add_trace(go.Scatter(x=tech_df.index, y=tech_df['%D'], name='%D', line=dict(color='#EF4444', dash='dot')))
        fig_stoch.add_hline(y=80, line_dash="dash", line_color="red")
        fig_stoch.add_hline(y=20, line_dash="dash", line_color="green")
        fig_stoch.update_layout(title="Stochastic Oscillator", height=300, template="plotly_dark")
        st.plotly_chart(fig_stoch, use_container_width=True, key=f"chart_stoch_{title}")
        
    with subtab_details:
        st.subheader("Comprehensive Strategy Report")
        
        # Risk Metrics
        var_95 = np.percentile(rets, 5)
        cvar_95 = rets[rets <= var_95].mean()
        skew = rets.skew()
        kurt = rets.kurtosis()
        
        # Ratios
        sortino = (ann_ret - 0.02) / (rets[rets < 0].std() * np.sqrt(12)) if rets[rets < 0].std() > 0 else 0
        calmar = ann_ret / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Stats
        win_rate = (rets > 0).mean()
        best_month = rets.max()
        worst_month = rets.min()
        
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            st.markdown("### 🛠️ Configuration & Inputs")
            cfg = loader.config
            
            # Determine Risk Aversion / Max Turnover from session state or defaults
            # Since these are local vars in run_backtest, we might not have them easily unless stored in results
            # But we can try to infer or use placeholders if not available.
            # Actually, 'results' dict doesn't store input params.
            # We should probably store them in results during run_backtest.
            # For now, I'll use placeholders or try to get from session state if active.
            
            # Helper to safely get param
            def get_param(key, default):
                if 'last_run_params' in st.session_state:
                    return st.session_state.last_run_params.get('optimizer_params', {}).get(key, default)
                return default

            # Get Max Drawdown Date
            mdd_date = drawdown_series.idxmin().date()
            
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
                    "N/A (MinRisk)", # Risk Aversion not applicable for MinVar/MinCVaR usually, or implied
                    f"{get_param('max_turnover', 'N/A')}",
                    f"{turnover:.2%}",
                    f"{cost_bps*10000:.0f} bps",
                    "Dynamic (FF3)" if use_dynamic_rf else "Fixed (0%)",
                    f"{get_param('cov_method', 'ledoit')}",
                    f"{get_param('solver', 'N/A')}",
                    f"{mdd_date}"
                ]
            }
            st.table(pd.DataFrame(config_data).set_index("Parameter"))
            
            if 'sector_constraints' in st.session_state and st.session_state.sector_constraints:
                st.markdown("**Sector Constraints:**")
                st.json(st.session_state.sector_constraints)

        with col_d2:
            st.markdown("### 📊 Advanced Performance Metrics")
            
            # Calculate cumulative return
            cum_ret = (1 + rets).prod() - 1
            
            perf_data = {
                "Metric": [
                    "Cumulative Return", "Annualized Return", "Annualized Volatility",
                    "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
                    "Max Drawdown", "Value at Risk (95%)", "CVaR (95%)",
                    "Skewness", "Kurtosis", "Win Rate (Months)",
                    "Best Month", "Worst Month", "Avg Turnover"
                ],
                "Value": [
                    f"{values.iloc[-1]:.2f}x", f"{ann_ret:.2%}", f"{ann_vol:.2%}",
                    f"{sharpe:.2f}", f"{sortino:.2f}", f"{calmar:.2f}",
                    f"{max_drawdown:.2%}", f"{var_95:.2%}", f"{cvar_95:.2%}",
                    f"{skew:.2f}", f"{kurt:.2f}", f"{win_rate:.1%}",
                    f"{best_month:.2%}", f"{worst_month:.2%}", f"{turnover:.2%}"
                ]
            }
            st.table(pd.DataFrame(perf_data).set_index("Metric"))
            
    with subtab_export:
        st.subheader("📥 Export Results")
        
        # Convert values to CSV
        csv_values = values.to_csv()
        st.download_button(
            label="Download Portfolio Values (CSV)",
            data=csv_values,
            file_name="portfolio_values.csv",
            mime="text/csv",
            key=f"dl_csv_val_{title}"
        )
        
        # Convert weights to CSV
        # Weights history is a list of Series, need to concat
        if weights_history:
            weights_df = pd.DataFrame(weights_history, index=results['dates'])
            csv_weights = weights_df.to_csv()
            st.download_button(
                label="Download Portfolio Weights (CSV)",
                data=csv_weights,
                file_name="portfolio_weights.csv",
                mime="text/csv",
                key=f"dl_csv_w_{title}"
            )

# --- MAIN TABS ---
tab_opt, tab_resamp, tab_frontier = st.tabs(["📊 Optimization Results", "✨ Resampling", "📉 Efficient Frontier"])

with tab_opt:
    if 'risk_results' in st.session_state:
        display_risk_results(st.session_state.risk_results, st.session_state.loader, "Base Optimization")
    else:
        st.info("Run optimization from the sidebar to see results.")

with tab_resamp:
    st.markdown("### Michaud Resampling")
    st.info("Resampling runs multiple simulations with perturbed inputs to find a more robust portfolio.")
    
    if st.session_state.loader is None:
        st.warning("Please load data first.")
    else:
        # Slider 1-10 (Explicit)
        n_sims = st.slider("Number of Simulations", 1, 10, 2, key="n_sims_risk")
        st.caption(f"Running {n_sims} simulations.")
        
        if st.button("▶️ Run Resampled Optimization", key="btn_resamp_risk"):
            if 'last_run_params' not in st.session_state:
                st.error("Please run the base optimization first to set up parameters.")
            else:
                with st.spinner(f"Running {n_sims} simulations..."):
                    params = st.session_state.last_run_params
                    
                    # Re-run with Resampling Enabled
                    results = run_backtest(
                        loader=st.session_state.loader,
                        optimizer_class=params['optimizer_class'],
                        optimizer_params=params['optimizer_params'],
                        sector_constraints=sector_constraints, # Live constraints from sidebar
                        cost_bps=cost_bps,
                        use_dynamic_rf=use_dynamic_rf,
                        optimize_kwargs=params['optimize_kwargs'],
                        benchmark_series=params['benchmark_series'],
                        enable_resampling=True,
                        n_simulations=n_sims
                    )
                    
                    # Add benchmarks (copy from base results)
                    if 'risk_results' in st.session_state:
                        results['bench_ew'] = st.session_state.risk_results.get('bench_ew')
                        results['bench_vw'] = st.session_state.risk_results.get('bench_vw')
                        results['active_benchmark_type'] = st.session_state.risk_results.get('active_benchmark_type')
                    
                    st.session_state.risk_resamp_results = results
                    st.success("Resampling Complete!")

    if 'risk_resamp_results' in st.session_state:
        st.markdown("---")
        # Pass base results as benchmark for comparison
        base_results = st.session_state.get('risk_results')
        display_risk_results(st.session_state.risk_resamp_results, st.session_state.loader, "Resampled Results", benchmark_results=base_results)

with tab_frontier:
    st.markdown("### 📉 Efficient Frontier Analysis")
    st.info("Calculates the static Efficient Frontier based on the entire selected history. This helps visualize the Risk/Return trade-off available in your universe.")
    
    col_f1, col_f2 = st.columns([1, 3])
    
    with col_f1:
        n_points = st.number_input("Number of Points", min_value=10, max_value=100, value=50, step=10)
        calc_btn = st.button("Calculate Frontier", type="primary")
    
    if calc_btn:
        with st.spinner("Calculating Efficient Frontier (solving 50+ optimizations)..."):
            if st.session_state.loader is None:
                st.error("Please load data first.")
            else:
                frontier_data = calculate_efficient_frontier(st.session_state.loader, points=n_points, solver=solver)
            
            if frontier_data:
                # Plot
                fig = go.Figure()
                
                # 1. Frontier Line
                fig.add_trace(go.Scatter(
                    x=frontier_data['frontier_vol'],
                    y=frontier_data['frontier_ret'],
                    mode='lines',
                    name='Efficient Frontier',
                    line=dict(color='#3B82F6', width=3)
                ))
                
                # 2. Assets Scatter
                fig.add_trace(go.Scatter(
                    x=frontier_data['assets_vol'],
                    y=frontier_data['assets_ret'],
                    mode='markers',
                    name='Assets',
                    text=frontier_data['assets_names'],
                    marker=dict(color='#9CA3AF', size=5, opacity=0.6)
                ))
                
                # 3. Highlight Max Sharpe (Approximate from Frontier)
                sharpes = np.array(frontier_data['frontier_ret']) / np.array(frontier_data['frontier_vol'])
                idx_max_sharpe = np.argmax(sharpes)
                
                fig.add_trace(go.Scatter(
                    x=[frontier_data['frontier_vol'][idx_max_sharpe]],
                    y=[frontier_data['frontier_ret'][idx_max_sharpe]],
                    mode='markers+text',
                    name='Max Sharpe',
                    text=['Max Sharpe'],
                    textposition="top left",
                    marker=dict(color='#EF4444', size=12, symbol='star')
                ))
                
                # 4. Highlight Min Vol
                idx_min_vol = np.argmin(frontier_data['frontier_vol'])
                
                fig.add_trace(go.Scatter(
                    x=[frontier_data['frontier_vol'][idx_min_vol]],
                    y=[frontier_data['frontier_ret'][idx_min_vol]],
                    mode='markers+text',
                    name='Min Volatility',
                    text=['Min Vol'],
                    textposition="bottom right",
                    marker=dict(color='#10B981', size=12, symbol='diamond')
                ))
                
                # 5. Current Portfolio (if available)
                if 'risk_results' in st.session_state:
                    curr_res = st.session_state.risk_results
                    # Calculate annualized ret/vol for the strategy
                    curr_rets = curr_res['values'].pct_change().dropna()
                    curr_ann_ret = curr_rets.mean() * 12
                    curr_ann_vol = curr_rets.std() * np.sqrt(12)
                    
                    fig.add_trace(go.Scatter(
                        x=[curr_ann_vol],
                        y=[curr_ann_ret],
                        mode='markers+text',
                        name='Current Strategy',
                        text=['Strategy'],
                        textposition="top center",
                        marker=dict(color='#F59E0B', size=14, symbol='cross')
                    ))

                fig.update_layout(
                    title="Efficient Frontier vs Assets",
                    xaxis_title="Annualized Volatility",
                    yaxis_title="Annualized Return",
                    template="plotly_dark",
                    height=600,
                    xaxis=dict(tickformat=".1%"),
                    yaxis=dict(tickformat=".1%")
                )
                st.plotly_chart(fig, use_container_width=True)


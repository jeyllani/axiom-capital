
import pandas as pd
import numpy as np
from src.portfolio_engine.data_loader import PortfolioDataLoader
from src.portfolio_engine.optimizers import MinVarianceOptimizer, MaxSharpeOptimizer, RiskParityOptimizer
from src.portfolio_engine.utils import get_optimal_solver
import streamlit as st

# Cache the loader to avoid reloading data on every page switch
@st.cache_resource
def get_shared_loader():
    config = {
        'data_dir': '../data/yfinance',
        'input_file': 'financial_universe_clean.parquet',
        'test_start': '2010-01-01',
        'test_end': '2024-12-31',
        'lookback_months': 24,
        'source': 'YFINANCE',
        'n_stocks': 50,
        'selection_method': 'random',
        'random_seed': 42
    }
    loader = PortfolioDataLoader(config)
    loader.load_data()
    return loader

def run_strategy(strategy_name):
    """
    Runs a specific strategy and returns the results.
    """
    loader = get_shared_loader()
    returns, caps, sector_map, name_map = loader.get_matrices()
    
    solver = get_optimal_solver()
    
    # Define Strategy
    if strategy_name == "Defensive MinVol":
        optimizer = MinVarianceOptimizer(risk_aversion=None, solver=solver)
        params = {}
    elif strategy_name == "Defensive Conservative":
        optimizer = MinVarianceOptimizer(risk_aversion=10.0, solver=solver)
        params = {}
    elif strategy_name == "Defensive Moderate":
        optimizer = MinVarianceOptimizer(risk_aversion=7.0, solver=solver)
        params = {}
    elif strategy_name == "Balanced Sharpe":
        optimizer = MaxSharpeOptimizer(solver=solver)
        params = {}
    elif strategy_name == "Balanced Growth":
        optimizer = MinVarianceOptimizer(risk_aversion=3.0, solver=solver)
        params = {}
    elif strategy_name == "Balanced RiskParity":
        optimizer = RiskParityOptimizer(solver=solver)
        params = {}
    elif strategy_name == "Aggressive Dynamic":
        optimizer = MinVarianceOptimizer(risk_aversion=1.0, solver=solver)
        params = {'max_asset_weight': 0.15}
    elif strategy_name == "Aggressive HighOctane":
        optimizer = MinVarianceOptimizer(risk_aversion=0.5, solver=solver)
        params = {'max_asset_weight': 0.15}
    elif strategy_name == "Aggressive MaxRet":
        optimizer = MinVarianceOptimizer(risk_aversion=0.1, solver=solver)
        params = {'max_asset_weight': 0.15}
    elif strategy_name == "ESG Impact":
        # ESG Constraints: No Energy, Utilities, or Materials (Proxy for "Dirty" sectors)
        esg_constraints = {
            'Energy': (0.0, 0.0),
            'Utilities': (0.0, 0.0),
            'Materials': (0.0, 0.0),
            'Industrials': (0.0, 0.0)
        }
        optimizer = MaxSharpeOptimizer(solver=solver)
        params = {'sector_constraints': esg_constraints, 'sector_map': loader.sector_map}
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
        
    # Run Backtest (Simplified for speed, similar to notebook)
    # We'll do a quarterly rebalance backtest
    
    start_date = pd.Timestamp(loader.config['test_start'])
    trading_dates = returns.index[returns.index >= start_date]
    
    # Rebalancing Frequency (Monthly for all strategies)
    rebalance_freq = 1
    
    portfolio_rets = []
    weights_history = [] # Store weights for allocation view
    dates_history = []
    
    prev_w = None
    current_weights = pd.Series(0, index=returns.columns)
    
    # Pre-calculate history start indices to speed up? No, just loop.
    
    progress_bar = st.progress(0)
    total_steps = len(trading_dates)
    
    for i, date in enumerate(trading_dates):
        # Update Progress
        progress_bar.progress((i + 1) / total_steps)
        
        # Rebalance
        if i % rebalance_freq == 0:
            history_start = date - pd.DateOffset(months=24)
            hist_rets = returns.loc[history_start:date].iloc[:-1]
            
            if len(hist_rets) > 12:
                try:
                    w = optimizer.optimize(hist_rets, prev_weights=prev_w, **params)
                    
                    # [FIX] Apply Max Asset Weight Fallback (Diversification)
                    # This is done here to avoid modifying the core optimizer classes.
                    if 'max_asset_weight' in params:
                        w = _apply_max_weight_fallback(w, params['max_asset_weight'])
                        
                    current_weights = w
                    prev_w = w
                    weights_history.append(w)
                    dates_history.append(date)
                except Exception:
                    pass
        
        # Calculate Return
        ret = (current_weights * returns.loc[date]).sum()
        portfolio_rets.append(ret)
        
    progress_bar.empty()
        
    # Create Result Series
    res_series = pd.Series(portfolio_rets, index=trading_dates)
    
    # --- Context Injection for Chatbot ---
    try:
        # Calculate Metrics
        ann_ret = res_series.mean() * 12
        ann_vol = res_series.std() * np.sqrt(12)
        sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
        
        # Drawdown
        cum = (1 + res_series).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        max_dd = dd.min()
        
        # Top Holdings
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
            "Page": f"Product: {strategy_name}",
            "Annual Return": f"{ann_ret:.2%}",
            "Volatility": f"{ann_vol:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd:.2%}",
            "Top Holdings": top_holdings_str
        }
        st.session_state['latest_optimization_context'] = context_data
    except Exception as e:
        print(f"Context Injection Error: {e}")

    # Return dictionary with everything needed for UI
    return {
        'values': (1 + res_series).cumprod(), # Cumulative Value
        'returns': res_series,
        'weights_history': weights_history, # List of Series
        'loader': loader # For sector mapping
    }

def _apply_max_weight_fallback(weights, max_weight):
    """
    Enforce maximum asset weight constraint via iterative clipping and redistribution.
    """
    if max_weight is None or max_weight >= 1.0:
        return weights
        
    w = weights.copy()
    n_assets = len(w)
    
    # Safety check: is it mathematically possible?
    if max_weight * n_assets < 1.0 - 1e-6:
        return pd.Series(1.0/n_assets, index=w.index)

    for _ in range(50):
        violators = w > max_weight + 1e-6
        if not violators.any(): break
        
        excess = (w[violators] - max_weight).sum()
        w[violators] = max_weight
        
        receivers = w < max_weight - 1e-6
        if not receivers.any(): break
            
        w_receivers = w[receivers]
        total_receiver_weight = w_receivers.sum()
        
        if total_receiver_weight > 1e-6:
            w[receivers] += excess * (w_receivers / total_receiver_weight)
        else:
            w[receivers] += excess / len(w_receivers)
            
        if w.sum() > 0: w = w / w.sum()
        
    return w

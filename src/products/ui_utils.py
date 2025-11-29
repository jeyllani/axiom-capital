
import streamlit as st
import pandas as pd
import numpy as np

def render_technical_specs(specs):
    """
    Renders a styled HTML table for technical specifications.
    specs: dict of {Label: Value}
    """
    rows = ""
    for k, v in specs.items():
        rows += f"""
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #334155; color: #94a3b8; font-weight: 600; width: 40%;">{k}</td>
            <td style="padding: 12px; border-bottom: 1px solid #334155; color: #f1f5f9;">{v}</td>
        </tr>
        """
    
    html = f"""
    <div style="background-color: rgba(30, 41, 59, 0.4); border: 1px solid #334155; border-radius: 12px; overflow: hidden;">
        <table style="width: 100%; border-collapse: collapse;">
            {rows}
        </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def calculate_drawdown(series):
    """Calculates the drawdown series."""
    hwm = series.cummax()
    drawdown = (series / hwm) - 1
    return drawdown

def render_performance_metrics(values):
    """
    Calculates and renders the Advanced Performance Metrics table.
    """
    # --- Advanced Metrics Calculation ---
    returns = values.pct_change().dropna()
    
    # Basic Metrics
    ann_ret = returns.mean() * 12
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
    
    # Drawdown
    drawdown_series = calculate_drawdown(values)
    max_drawdown = drawdown_series.min()
    max_dd_date = drawdown_series.idxmin().strftime('%Y-%m-%d')
    
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
    
    st.markdown("### 📊 Advanced Performance Metrics")
    
    perf_data = {
        "Metric": [
            "Annualized Return", "Annualized Volatility", "Sharpe Ratio", 
            "Sortino Ratio", "Calmar Ratio", "Max Drawdown", "Max Drawdown Date",
            "Value at Risk (95%)", "CVaR (95%)", "Skewness", "Kurtosis", 
            "Win Rate (Months)", "Best Month", "Worst Month"
        ],
        "Value": [
            f"{ann_ret:.2%}", f"{ann_vol:.2%}", f"{sharpe:.2f}",
            f"{sortino:.2f}", f"{calmar:.2f}", f"{max_drawdown:.2%}", f"{max_dd_date}",
            f"{var_95:.2%}", f"{cvar_95:.2%}", f"{skew:.2f}", f"{kurt:.2f}",
            f"{win_rate:.1%}", f"{best_month:.2%}", f"{worst_month:.2%}"
        ]
    }
    st.table(pd.DataFrame(perf_data).set_index("Metric"))

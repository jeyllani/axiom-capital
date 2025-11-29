
import streamlit as st

# Custom CSS for Advanced UI Design
st.markdown("""
<style>
    [data-testid="stSidebar"] {display: none;}
    
    /* Global Font */
    h1, h2, h3, p {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* ==========================================================================
       1. Back Button Styling (Top of page)
       ========================================================================== */
    div[data-testid="stVerticalBlock"] > div:first-child .stButton button {
        width: auto !important;
        padding: 8px 20px !important;
        background-color: transparent !important;
        border: 1px solid #475569 !important;
        color: #94a3b8 !important;
        border-radius: 20px !important;
        font-size: 14px !important;
        transition: all 0.2s ease;
    }
    div[data-testid="stVerticalBlock"] > div:first-child .stButton button:hover {
        border-color: #cbd5e1 !important;
        color: #f8fafc !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
    }

    /* ==========================================================================
       2. ESG Button (Primary)
       ========================================================================== */
    button[kind="primary"] {
        width: 100% !important;
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 78, 59, 0.15) 100%) !important;
        border: 1px solid #10b981 !important;
        border-radius: 16px !important;
        padding: 30px !important;
        text-align: left !important; /* Left aligned as requested */
        color: #cbd5e1 !important;
        white-space: pre-wrap !important;
        display: block !important;
        transition: all 0.3s ease !important;
        height: auto !important;
        min-height: 120px !important;
    }
    
    button[kind="primary"]::first-line {
        color: #10b981 !important;
        font-weight: 800 !important;
        font-size: 1.5rem !important;
        line-height: 2.5 !important;
    }
    
    button[kind="primary"]:hover {
        transform: scale(1.01) !important;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.25) !important;
        border-color: #34d399 !important;
        color: #ffffff !important;
    }

    /* ==========================================================================
       3. Product Grid Buttons (Inside Columns)
       ========================================================================== */
    
    /* Common Card Styling */
    [data-testid="column"] .stButton button {
        width: 100% !important;
        height: auto !important;
        min-height: 220px !important;
        white-space: pre-wrap !important;
        text-align: left !important;
        padding: 20px 24px 24px 24px !important; /* Reduced top padding */
        border-radius: 16px !important;
        border: 1px solid #334155 !important;
        background-color: rgba(30, 41, 59, 0.4) !important;
        color: #cbd5e1 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        display: block !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        line-height: 1.6 !important;
    }

    /* --- COLUMN 1: DEFENSIVE (Blue) --- */
    [data-testid="column"]:nth-of-type(1) .stButton button {
        border-top: 4px solid #38bdf8 !important;
        background: linear-gradient(180deg, rgba(56, 189, 248, 0.05) 0%, rgba(30, 41, 59, 0.4) 100%) !important;
    }
    [data-testid="column"]:nth-of-type(1) .stButton button::first-line {
        color: #38bdf8 !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
        line-height: 3.0 !important; /* Slightly reduced line-height */
    }
    [data-testid="column"]:nth-of-type(1) .stButton button:hover {
        transform: translateY(-5px) !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 10px 30px -5px rgba(56, 189, 248, 0.3) !important;
        color: #f1f5f9 !important;
    }

    /* --- COLUMN 2: BALANCED (Purple) --- */
    [data-testid="column"]:nth-of-type(2) .stButton button {
        border-top: 4px solid #a855f7 !important;
        background: linear-gradient(180deg, rgba(168, 85, 247, 0.05) 0%, rgba(30, 41, 59, 0.4) 100%) !important;
    }
    [data-testid="column"]:nth-of-type(2) .stButton button::first-line {
        color: #a855f7 !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
        line-height: 3.0 !important;
    }
    [data-testid="column"]:nth-of-type(2) .stButton button:hover {
        transform: translateY(-5px) !important;
        border-color: #a855f7 !important;
        box-shadow: 0 10px 30px -5px rgba(168, 85, 247, 0.3) !important;
        color: #f1f5f9 !important;
    }

    /* --- COLUMN 3: AGGRESSIVE (Orange) --- */
    [data-testid="column"]:nth-of-type(3) .stButton button {
        border-top: 4px solid #f97316 !important;
        background: linear-gradient(180deg, rgba(249, 115, 22, 0.05) 0%, rgba(30, 41, 59, 0.4) 100%) !important;
    }
    [data-testid="column"]:nth-of-type(3) .stButton button::first-line {
        color: #f97316 !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
        line-height: 3.0 !important;
    }
    [data-testid="column"]:nth-of-type(3) .stButton button:hover {
        transform: translateY(-5px) !important;
        border-color: #f97316 !important;
        box-shadow: 0 10px 30px -5px rgba(249, 115, 22, 0.3) !important;
        color: #f1f5f9 !important;
    }

    /* Category Headers */
    .category-header {
        font-size: 1.0rem;
        font-weight: 700;
        margin-bottom: 20px;
        padding-bottom: 5px;
        border-bottom: 2px solid #334155;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #94a3b8; /* Default fallback */
    }
    
    /* Specific Header Colors */
    .header-defensive { color: #38bdf8 !important; border-color: #38bdf8 !important; }
    .header-balanced { color: #a855f7 !important; border-color: #a855f7 !important; }
    .header-aggressive { color: #f97316 !important; border-color: #f97316 !important; }
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 50px 20px 30px;
    }
    
</style>
""", unsafe_allow_html=True)

# --- Navigation (Back Button) ---
if st.button("⬅️ Back to Home"):
    st.switch_page("pages/landing.py")

# --- Hero Section ---
st.markdown("""
<div class="hero-section">
    <h1 style="font-size: 2.8rem; font-weight: 300; margin-bottom: 10px; color: #f8fafc;">
        Axiom <span style="font-weight: 700; color: #94a3b8;">Capital</span>
    </h1>
    <p style="font-size: 1.2rem; color: #cbd5e1; font-weight: 300; letter-spacing: 0.02em;">
        Precision Engineering for Your Financial Legacy.
    </p>
</div>
""", unsafe_allow_html=True)

# --- ESG Spotlight ---
st.markdown("<h3 style='text-align: left; color: #10b981;'>🌱 Sustainable Alpha</h3>", unsafe_allow_html=True)

esg_text = "Global Sustainable Future\n\nAlign your wealth with your values.\n\nInvest in a better world without compromising returns. Zero carbon exposure with optimized risk-adjusted performance."

if st.button(esg_text, type="primary", use_container_width=True):
    st.switch_page("pages/products/esg_1.py")

st.markdown("###")

# --- Product Grid ---
col1, col2, col3 = st.columns(3, gap="medium")

# Helper to create styled card-buttons
def product_btn(title, tagline, desc, page, key):
    label = f"{title}\n\n{tagline}\n\n{desc}"
    if st.button(label, key=key, use_container_width=True):
        st.switch_page(page)

# --- Defensive Column (Column 1) ---
with col1:
    st.markdown('<div class="category-header header-defensive">🛡️ Defensive</div>', unsafe_allow_html=True)
    
    product_btn(
        "Global Low Volatility",
        "Stability in a volatile world.",
        "Systematic variance minimization for capital preservation.",
        "pages/products/defensive_1.py",
        "btn_def_1"
    )
    
    product_btn(
        "Global Conservative Yield",
        "Income with peace of mind.",
        "Optimized risk-adjusted income for the cautious investor.",
        "pages/products/defensive_2.py",
        "btn_def_2"
    )
    
    product_btn(
        "Global Moderate Growth",
        "The perfect balance of safety and growth.",
        "Balanced defense with controlled equity exposure for your capital.",
        "pages/products/defensive_3.py",
        "btn_def_3"
    )

# --- Balanced Column (Column 2) ---
with col2:
    st.markdown('<div class="category-header header-balanced">⚖️ Balanced</div>', unsafe_allow_html=True)
    
    product_btn(
        "Global Core Balanced",
        "Mathematically optimized for efficiency.",
        "Maximize Sharpe Ratio. The mathematical optimal portfolio.",
        "pages/products/balanced_1.py",
        "btn_bal_1"
    )
    
    product_btn(
        "Global Dynamic Growth",
        "Accelerate your wealth creation.",
        "Active capital appreciation for long-term wealth compounding.",
        "pages/products/balanced_2.py",
        "btn_bal_2"
    )
    
    product_btn(
        "Global Risk Parity",
        "True diversification for any climate.",
        "Equal Risk Contribution. True diversification across regimes.",
        "pages/products/balanced_3.py",
        "btn_bal_3"
    )

# --- Aggressive Column (Column 3) ---
with col3:
    st.markdown('<div class="category-header header-aggressive">🚀 Aggressive</div>', unsafe_allow_html=True)
    
    product_btn(
        "Global Dynamic Aggressive",
        "Unleash your portfolio's potential.",
        "High-momentum allocation for maximum capital expansion.",
        "pages/products/aggressive_1.py",
        "btn_agg_1"
    )
    
    product_btn(
        "Global High Octane",
        "Power and performance unleashed.",
        "Unconstrained growth targeting exponential returns for your capital.",
        "pages/products/aggressive_2.py",
        "btn_agg_2"
    )
    
    product_btn(
        "Global Max Return",
        "Limitless horizons for the bold.",
        "Absolute return focus for the highest risk tolerance for your capital.",
        "pages/products/aggressive_3.py",
        "btn_agg_3"
    )

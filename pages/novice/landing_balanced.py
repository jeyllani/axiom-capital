
import streamlit as st

# Custom CSS for Advanced UI Design (Balanced Theme)
st.markdown("""
<style>
    [data-testid="stSidebar"] {display: none;}
    
    /* Global Font */
    h1, h2, h3, p {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Back Button Styling */
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

    /* Balanced Card Styling */
    [data-testid="column"] .stButton button {
        width: 100% !important;
        height: auto !important;
        min-height: 220px !important;
        white-space: pre-wrap !important;
        text-align: left !important;
        padding: 24px !important;
        border-radius: 16px !important;
        border: 1px solid #334155 !important;
        background-color: rgba(30, 41, 59, 0.4) !important;
        color: #cbd5e1 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        display: block !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        line-height: 1.6 !important;
        
        /* Balanced Specifics */
        border-top: 4px solid #a855f7 !important; /* Purple */
        background: linear-gradient(180deg, rgba(168, 85, 247, 0.05) 0%, rgba(30, 41, 59, 0.4) 100%) !important;
    }
    
    [data-testid="column"] .stButton button::first-line {
        color: #a855f7 !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
        line-height: 3.5 !important;
    }
    
    [data-testid="column"] .stButton button:hover {
        transform: translateY(-5px) !important;
        border-color: #a855f7 !important;
        box-shadow: 0 10px 30px -5px rgba(168, 85, 247, 0.3) !important;
        color: #f1f5f9 !important;
    }
    
    /* Marketing Section Styling */
    .marketing-box {
        margin-top: 40px;
        padding: 30px;
        background: rgba(168, 85, 247, 0.05);
        border: 1px solid rgba(168, 85, 247, 0.2);
        border-radius: 16px;
        color: #cbd5e1;
    }
    .marketing-title {
        color: #a855f7;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* ESG Button (Primary) */
    button[kind="primary"] {
        width: 100% !important;
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 78, 59, 0.15) 100%) !important;
        border: 1px solid #10b981 !important;
        border-radius: 16px !important;
        padding: 30px !important;
        text-align: left !important;
        color: #cbd5e1 !important;
        white-space: pre-wrap !important;
        display: block !important;
        transition: all 0.3s ease !important;
        height: auto !important;
        min-height: 120px !important;
        margin-top: 40px !important;
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

</style>
""", unsafe_allow_html=True)

# Navigation
if st.button("⬅️ Back to Questionnaire"):
    st.switch_page("pages/novice/questionnaire.py")

st.title("⚖️ Balanced Strategies")
st.markdown("<h3 style='color: #94a3b8; font-weight: 300; margin-top: -10px;'>Optimal Risk-Reward Trade-off</h3>", unsafe_allow_html=True)
st.markdown("---")

# Product Grid
col1, col2, col3 = st.columns(3, gap="medium")

def product_btn(title, desc, page, key):
    label = f"{title}\n\n{desc}"
    if st.button(label, key=key, use_container_width=True):
        st.switch_page(page)

with col1:
    product_btn(
        "Global Core Balanced",
        "Flagship strategy maximizing Sharpe Ratio. The mathematical optimal portfolio (λ=2.0).",
        "pages/products/balanced_1.py",
        "btn_bal_1"
    )

with col2:
    product_btn(
        "Global Dynamic Growth",
        "Targets active capital appreciation with lower risk aversion for growth investors (λ=1.5).",
        "pages/products/balanced_2.py",
        "btn_bal_2"
    )

with col3:
    product_btn(
        "Global Risk Parity",
        "Allocates based on risk contribution. True diversification for all environments.",
        "pages/products/balanced_3.py",
        "btn_bal_3"
    )

# --- ESG Spotlight ---
st.markdown("<h3 style='text-align: left; color: #10b981; margin-top: 40px;'>🌱 Sustainable Alpha</h3>", unsafe_allow_html=True)

esg_text = "Global Sustainable Future\n\nInvest in a better world without compromising returns. Zero carbon exposure with optimized risk-adjusted performance."

if st.button(esg_text, type="primary", use_container_width=True):
    st.switch_page("pages/products/esg_1.py")

# Marketing Advice
st.markdown("""
<div class="marketing-box">
    <div class="marketing-title">
        <span>💡</span> The Power of Balance
    </div>
    <p style="font-size: 1.05rem; line-height: 1.6;">
        Achieve the perfect equilibrium between growth and stability. Our balanced strategies are engineered to capture 
        <b>market upside</b> while providing a robust cushion during downturns. This is the smart choice for medium to 
        long-term wealth preservation and growth, offering a smoother ride than pure equity strategies.
    </p>
    <p style="font-size: 1.05rem; line-height: 1.6; margin-top: 10px;">
        <i>"It is not timing the market, but time in the market that matters. A balanced approach keeps you invested through all cycles."</i>
    </p>
</div>
""", unsafe_allow_html=True)

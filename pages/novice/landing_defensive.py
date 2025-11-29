
import streamlit as st

# Custom CSS for Advanced UI Design (Defensive Theme)
st.markdown("""
<style>
    [data-testid="stSidebar"] {display: none;}
    
    /* Reduce top padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
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

    /* Defensive Card Styling */
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
        
        /* Defensive Specifics */
        border-top: 4px solid #38bdf8 !important; /* Blue */
        background: linear-gradient(180deg, rgba(56, 189, 248, 0.05) 0%, rgba(30, 41, 59, 0.4) 100%) !important;
    }
    
    [data-testid="column"] .stButton button::first-line {
        color: #38bdf8 !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
        line-height: 3.5 !important;
    }
    
    [data-testid="column"] .stButton button:hover {
        transform: translateY(-5px) !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 10px 30px -5px rgba(56, 189, 248, 0.3) !important;
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
        color: #38bdf8; /* Blue */
        border-color: #38bdf8;
    }
    
    /* Marketing Section Styling */
    .marketing-box {
        margin-top: 40px;
        padding: 40px;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.6) 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        color: #cbd5e1;
        text-align: center;
    }
    .marketing-title {
        color: #38bdf8;
        font-size: 1.5rem;
        font-weight: 300;
        margin-bottom: 20px;
        letter-spacing: 0.05em;
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
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 0px 20px 40px;
    }

</style>
""", unsafe_allow_html=True)

# Navigation
if st.button("⬅️ Back to Questionnaire"):
    st.switch_page("pages/novice/questionnaire.py")

# Hero Section
st.markdown("""
<div class="hero-section">
    <h1 style="font-size: 2.8rem; font-weight: 300; margin-bottom: 10px; color: #f8fafc;">
        Defensive <span style="font-weight: 700; color: #38bdf8;">Strategies</span>
    </h1>
    <p style="font-size: 1.2rem; color: #cbd5e1; font-weight: 300; letter-spacing: 0.02em;">
        Capital Preservation & Stability.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("###")

# Product Grid
col1, col2, col3 = st.columns(3, gap="medium")

def product_btn(title, tagline, desc, page, key):
    label = f"{title}\n\n{tagline}\n\n{desc}"
    if st.button(label, key=key, use_container_width=True):
        st.switch_page(page)

with col1:
    st.markdown('<div class="category-header">🛡️ Low Volatility</div>', unsafe_allow_html=True)
    product_btn(
        "Global Low Volatility",
        "Stability First.",
        "Systematic variance minimization for capital preservation. Lowest risk profile.",
        "pages/products/defensive_1.py",
        "btn_def_1"
    )

with col2:
    st.markdown('<div class="category-header">💰 Conservative Yield</div>', unsafe_allow_html=True)
    product_btn(
        "Global Conservative Yield",
        "Income & Safety.",
        "Optimized risk-adjusted income for the cautious investor. High risk aversion (λ=10).",
        "pages/products/defensive_2.py",
        "btn_def_2"
    )

with col3:
    st.markdown('<div class="category-header">📈 Moderate Growth</div>', unsafe_allow_html=True)
    product_btn(
        "Global Moderate Growth",
        "Balanced Defense.",
        "Balanced defense with controlled equity exposure. Moderate risk aversion (λ=7).",
        "pages/products/defensive_3.py",
        "btn_def_3"
    )

# --- ESG Spotlight ---
st.markdown("<h3 style='text-align: left; color: #10b981; margin-top: 40px;'>🌱 Sustainable Alpha</h3>", unsafe_allow_html=True)

esg_text = "Global Sustainable Future\n\nAlign your wealth with your values.\n\nInvest in a better world without compromising returns. Zero carbon exposure with optimized risk-adjusted performance."

if st.button(esg_text, type="primary", use_container_width=True):
    st.switch_page("pages/products/esg_1.py")

# Marketing Advice
st.markdown("""
<div class="marketing-box">
    <div class="marketing-title">
        <span>💡</span> Preserve and Grow
    </div>
    <p style="font-size: 1.1rem; line-height: 1.8; max-width: 800px; margin: 0 auto;">
        Prioritize capital protection without sacrificing returns. Our defensive strategies utilize advanced 
        <b>variance minimization techniques</b> to deliver steady, reliable growth. This is the perfect solution for 
        conservative investors, those approaching retirement, or anyone who values a good night's sleep over market noise.
    </p>
    <p style="font-size: 1.1rem; line-height: 1.8; margin-top: 15px; color: #94a3b8; font-style: italic;">
        "Rule No. 1: Never lose money. Rule No. 2: Never forget rule No. 1." - Warren Buffett
    </p>
</div>
""", unsafe_allow_html=True)

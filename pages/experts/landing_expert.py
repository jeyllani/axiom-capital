
import streamlit as st

# Custom CSS for Advanced UI Design (Expert Theme)
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
       2. Product Grid Buttons (Inside Columns)
       ========================================================================== */
    
    /* Common Card Styling */
    [data-testid="column"] .stButton button,
    [data-testid="column"] button,
    [data-testid="stColumn"] .stButton button,
    [data-testid="stColumn"] button {
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

    /* --- COLUMN 1: UTILITY (Indigo) --- */
    [data-testid="column"]:nth-of-type(1) .stButton button,
    [data-testid="column"]:nth-of-type(1) button,
    [data-testid="stColumn"]:nth-of-type(1) .stButton button,
    [data-testid="stColumn"]:nth-of-type(1) button {
        border-top: 4px solid #6366f1 !important; /* Indigo */
        background: linear-gradient(180deg, rgba(99, 102, 241, 0.05) 0%, rgba(30, 41, 59, 0.4) 100%) !important;
    }
    [data-testid="column"]:nth-of-type(1) .stButton button::first-line,
    [data-testid="column"]:nth-of-type(1) button::first-line,
    [data-testid="stColumn"]:nth-of-type(1) .stButton button::first-line,
    [data-testid="stColumn"]:nth-of-type(1) button::first-line {
        color: #6366f1 !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
        line-height: 3.0 !important;
    }
    [data-testid="column"]:nth-of-type(1) .stButton button:hover {
        transform: translateY(-5px) !important;
        border-color: #6366f1 !important;
        box-shadow: 0 10px 30px -5px rgba(99, 102, 241, 0.3) !important;
        color: #f1f5f9 !important;
    }

    /* --- COLUMN 2: RISK (Rose) --- */
    [data-testid="column"]:nth-of-type(2) .stButton button,
    [data-testid="column"]:nth-of-type(2) button,
    [data-testid="stColumn"]:nth-of-type(2) .stButton button,
    [data-testid="stColumn"]:nth-of-type(2) button {
        border-top: 4px solid #f43f5e !important; /* Rose */
        background: linear-gradient(180deg, rgba(244, 63, 94, 0.05) 0%, rgba(30, 41, 59, 0.4) 100%) !important;
    }
    [data-testid="column"]:nth-of-type(2) .stButton button::first-line,
    [data-testid="column"]:nth-of-type(2) button::first-line,
    [data-testid="stColumn"]:nth-of-type(2) .stButton button::first-line,
    [data-testid="stColumn"]:nth-of-type(2) button::first-line {
        color: #f43f5e !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
        line-height: 3.0 !important;
    }
    [data-testid="column"]:nth-of-type(2) .stButton button:hover {
        transform: translateY(-5px) !important;
        border-color: #f43f5e !important;
        box-shadow: 0 10px 30px -5px rgba(244, 63, 94, 0.3) !important;
        color: #f1f5f9 !important;
    }

    /* --- COLUMN 3: FRONTIER (Cyan) --- */
    [data-testid="column"]:nth-of-type(3) .stButton button,
    [data-testid="column"]:nth-of-type(3) button,
    [data-testid="stColumn"]:nth-of-type(3) .stButton button,
    [data-testid="stColumn"]:nth-of-type(3) button {
        border-top: 4px solid #06b6d4 !important; /* Cyan */
        background: linear-gradient(180deg, rgba(6, 182, 212, 0.05) 0%, rgba(30, 41, 59, 0.4) 100%) !important;
    }
    [data-testid="column"]:nth-of-type(3) .stButton button::first-line,
    [data-testid="column"]:nth-of-type(3) button::first-line,
    [data-testid="stColumn"]:nth-of-type(3) .stButton button::first-line,
    [data-testid="stColumn"]:nth-of-type(3) button::first-line {
        color: #06b6d4 !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
        line-height: 3.0 !important;
    }
    [data-testid="column"]:nth-of-type(3) .stButton button:hover {
        transform: translateY(-5px) !important;
        border-color: #06b6d4 !important;
        box-shadow: 0 10px 30px -5px rgba(6, 182, 212, 0.3) !important;
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
    .header-utility { color: #6366f1 !important; border-color: #6366f1 !important; }
    .header-risk { color: #f43f5e !important; border-color: #f43f5e !important; }
    .header-frontier { color: #06b6d4 !important; border-color: #06b6d4 !important; }

    /* Marketing Section Styling */
    .marketing-box {
        margin-top: 50px;
        padding: 40px;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.6) 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        color: #cbd5e1;
        text-align: center;
    }
    .marketing-title {
        color: #f8fafc;
        font-size: 1.5rem;
        font-weight: 300;
        margin-bottom: 20px;
        letter-spacing: 0.05em;
    }
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 0px 20px 40px; /* Reduced top padding from 50px */
    }

</style>
""", unsafe_allow_html=True)

# Navigation
if st.button("⬅️ Back to Home"):
    st.switch_page("pages/landing.py")

# Hero Section
st.markdown("""
<div class="hero-section">
    <h1 style="font-size: 2.8rem; font-weight: 300; margin-bottom: 10px; color: #f8fafc;">
        Axiom <span style="font-weight: 700; color: #94a3b8;">Quantitative Framework</span>
    </h1>
    <p style="font-size: 1.2rem; color: #cbd5e1; font-weight: 300; letter-spacing: 0.02em;">
        Advanced Engineering for Portfolio Optimization and Risk Management.
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

# --- Utility Column ---
with col1:
    st.markdown('<div class="category-header header-utility">📐 Optimization</div>', unsafe_allow_html=True)
    product_btn(
        "Utility Maximization",
        "Tailored Risk Profiles.",
        "Construct portfolios tailored to specific risk aversion profiles (λ). Use Michaud Resampling for robust, stable allocations.",
        "pages/experts/expert_utility.py",
        "btn_utility"
    )

# --- Risk Column ---
with col2:
    st.markdown('<div class="category-header header-risk">🛡️ Risk Architecture</div>', unsafe_allow_html=True)
    product_btn(
        "Risk Analytics",
        "Deep Risk Decomposition.",
        "Deep dive into portfolio risk decomposition. Analyze Component VaR, Marginal Contribution to Risk, and stress test scenarios.",
        "pages/experts/expert_risk.py",
        "btn_risk"
    )

# --- Frontier Column ---
with col3:
    st.markdown('<div class="category-header header-frontier">📈 Efficient Frontier</div>', unsafe_allow_html=True)
    product_btn(
        "Frontier Visualization",
        "Market Universe Mapping.",
        "Map the full investment universe. Visualize the Efficient Frontier to benchmark your strategy against the theoretical mathematical optimums.",
        "pages/experts/expert_frontier.py",
        "btn_frontier"
    )

# Marketing Advice / Framework Philosophy
st.markdown("""
<div class="marketing-box">
    <div class="marketing-title">
        The Axiom Advantage
    </div>
    <p style="font-size: 1.1rem; line-height: 1.8; max-width: 800px; margin: 0 auto;">
        The <b>Axiom Quantitative Framework</b> empowers investment professionals to move beyond simple heuristics. 
        By rigorously applying Modern Portfolio Theory (MPT) and advanced risk budgeting, we enable the construction of 
        portfolios that are mathematically robust and aligned with precise investment mandates.
    </p>
    <div style="display: flex; justify-content: center; gap: 40px; margin-top: 30px;">
        <div>
            <div style="font-size: 1.5rem; margin-bottom: 5px;">🎯</div>
            <div style="color: #94a3b8; font-size: 0.9rem;">Precise Targeting</div>
        </div>
        <div>
            <div style="font-size: 1.5rem; margin-bottom: 5px;">🔍</div>
            <div style="color: #94a3b8; font-size: 0.9rem;">Granular Analysis</div>
        </div>
        <div>
            <div style="font-size: 1.5rem; margin-bottom: 5px;">⚡</div>
            <div style="color: #94a3b8; font-size: 0.9rem;">Robust Execution</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

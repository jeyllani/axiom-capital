import streamlit as st

# Custom CSS for Axiom Capital Design
st.markdown("""
<style>
    [data-testid="stSidebar"] {display: none;}
    
    /* Reduce top padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Global Font */
    h1, h2, h3, p, div {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Hero Section - Clean (No Box) */
    .hero-container {
        text-align: center;
        padding: 0px 20px 20px; /* Reduced top padding from 40px */
        margin-bottom: 40px;
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 200; /* Thinner, more elegant */
        color: #f8fafc;
        margin-bottom: 10px;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: #94a3b8;
        font-weight: 300;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        border-top: 1px solid #334155;
        display: inline-block;
        padding-top: 10px;
        margin-top: 10px;
    }
    
    /* Profile Cards (Clickable Areas) */
    .profile-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 30px;
        height: 100%;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        text-align: center;
        cursor: pointer;
    }
    
    .profile-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.3);
        background: rgba(30, 41, 59, 0.6);
    }
    
    .card-icon {
        font-size: 3rem;
        margin-bottom: 15px;
    }
    
    .card-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .card-sub {
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 300;
    }

    /* Marketing Content Section */
    .marketing-section {
        margin-top: 50px;
        padding: 30px;
        border-top: 1px solid #1e293b;
    }
    
    .marketing-col {
        padding: 10px;
    }
    
    .marketing-title {
        color: #38bdf8;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    .marketing-text {
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* FAQ Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(30, 41, 59, 0.4);
        border-radius: 8px 8px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #94a3b8;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(15, 23, 42, 0.8);
        color: #38bdf8;
        border-bottom: 2px solid #38bdf8;
    }
    
    /* Specific Card Themes */
    .theme-advisory { border-bottom: 4px solid #10b981; }
    .theme-funds { border-bottom: 4px solid #3b82f6; }
    .theme-desk { border-bottom: 4px solid #f97316; }

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

</style>
""", unsafe_allow_html=True)

# --- Navigation (Back Button) ---
if st.button("⬅️ Back to Home"):
    st.switch_page("pages/index.py")

# --- Hero Section (Clean) ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Axiom <span style="font-weight: 700; color: #f8fafc;">Capital</span></div>
    <div class="hero-subtitle">Precision Engineering for Your Financial Legacy</div>
</div>
""", unsafe_allow_html=True)

# --- Profile Selection (The Choice) ---
col1, col2, col3 = st.columns(3, gap="medium")

# 1. Advisory (Novice)
with col1:
    st.markdown("""
    <div class="profile-card theme-advisory">
        <div class="card-icon">🧭</div>
        <div class="card-title">Wealth Advisory</div>
        <div class="card-sub">Guided Portfolio Construction</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Enter Advisory", key="btn_novice", use_container_width=True):
        st.switch_page("pages/novice/questionnaire.py")

# 2. Funds (Intermediate)
with col2:
    st.markdown("""
    <div class="profile-card theme-funds">
        <div class="card-icon">🏛️</div>
        <div class="card-title">Strategic Funds</div>
        <div class="card-sub">Curated Investment Catalog</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Browse Funds", key="btn_products", use_container_width=True):
        st.switch_page("pages/products/landing_products.py")

# 3. Desk (Expert)
with col3:
    st.markdown("""
    <div class="profile-card theme-desk">
        <div class="card-icon">⚡</div>
        <div class="card-title">Quantitative Desk</div>
        <div class="card-sub">Advanced Analytics Terminal</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Launch Terminal", key="btn_expert", use_container_width=True):
        st.switch_page("pages/experts/landing_expert.py")

# --- Marketing Explanatory Content ---
st.markdown("<div class='marketing-section'></div>", unsafe_allow_html=True)

m_col1, m_col2, m_col3 = st.columns(3, gap="medium")

with m_col1:
    st.markdown("""
    <div class="marketing-col">
        <div class="marketing-title" style="color: #10b981;">For Private Clients</div>
        <div class="marketing-text">
            Not sure where to start? Our <b>Wealth Advisory</b> service uses a behavioral finance questionnaire to map your risk tolerance to the optimal portfolio. 
            We handle the complexity of asset allocation so you can focus on your life goals.
        </div>
    </div>
    """, unsafe_allow_html=True)

with m_col2:
    st.markdown("""
    <div class="marketing-col">
        <div class="marketing-title" style="color: #3b82f6;">For Asset Allocators</div>
        <div class="marketing-text">
            Build your own legacy with our <b>Strategic Funds</b>. 
            Access a transparent catalog of institutional-grade strategies, from "Global Low Volatility" to "High Octane Growth". 
            Each fund is rigorously backtested and optimized for specific outcomes.
        </div>
    </div>
    """, unsafe_allow_html=True)

with m_col3:
    st.markdown("""
    <div class="marketing-col">
        <div class="marketing-title" style="color: #f97316;">For Professional Traders</div>
        <div class="marketing-text">
            The <b>Quantitative Desk</b> is your sandbox. 
            Unlock the full power of the Axiom engine: customize risk aversion (λ), visualize the Efficient Frontier in real-time, and perform deep-dive stress testing on custom portfolios.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- Comprehensive FAQ (Tabs) ---
st.markdown("<h3 style='text-align: center; margin-bottom: 30px; color: #f8fafc;'>Knowledge Center</h3>", unsafe_allow_html=True)

tab_gen, tab_adv, tab_strat, tab_quant = st.tabs(["🏛️ Philosophy", "🧭 Advisory", "📈 Strategies", "🔬 Analytics"])

with tab_gen:
    st.markdown("""
    #### Why Axiom Capital?
    We believe that **mathematics beats speculation**. In a world of noise, we rely on signal. Our investment philosophy is grounded in Modern Portfolio Theory (MPT) and advanced risk management techniques.
    
    **Core Principles:**
    *   **Diversification is the only free lunch**: We don't just diversify across assets, but across risk factors.
    *   **Risk comes first**: We define the risk budget first, then maximize returns within that constraint.
    *   **Transparency**: No black boxes. We provide full visibility into our metrics and methodology.
    """)

with tab_adv:
    st.markdown("""
    #### How does the Advisory process work?
    Our **Wealth Advisory** module is designed to bridge the gap between your psychological risk tolerance and your financial capacity for loss.
    
    1.  **Assessment**: You answer a series of psychometric questions.
    2.  **Scoring**: Our algorithm calculates a composite Risk Score (0-100).
    3.  **Matching**: We map your score to one of our three core profiles:
        *   **Defensive**: Focus on capital preservation (MinVol).
        *   **Balanced**: Focus on risk-adjusted returns (Max Sharpe).
        *   **Aggressive**: Focus on capital appreciation (Utility Max).
    """)

with tab_strat:
    st.markdown("""
    #### Understanding Our Fund Shelf
    We offer a spectrum of strategies tailored to different market views and objectives.
    
    *   **Defensive Series**: Built on **Global Minimum Variance** optimization. These strategies seek to construct the portfolio with the lowest possible volatility, offering stability during market turbulence.
    *   **Balanced Series**: Built on **Maximum Sharpe Ratio** (Tangency Portfolio) and **Risk Parity** (ERC). These aim for the highest return per unit of risk.
    *   **Aggressive Series**: Built on **Utility Maximization** with low risk aversion coefficients. These portfolios tilt towards high-beta assets to capture maximum upside during bull markets.
    *   **ESG Impact**: A constrained optimization that excludes non-compliant sectors while maintaining the mathematical rigor of our standard portfolios.
    """)

with tab_quant:
    st.markdown("""
    #### The Quantitative Desk
    This is where the engine room is exposed.
    
    *   **Efficient Frontier**: Visualize the set of optimal portfolios that offer the highest expected return for a defined level of risk.
    *   **Monte Carlo Simulation**: We use stochastic calculus (Geometric Brownian Motion) to project thousands of potential future price paths, giving you a probabilistic view of future wealth.
    *   **Resampling**: We employ Michaud Resampling techniques to address estimation error in the covariance matrix, resulting in more robust and diversified portfolios.
    """)

st.markdown("""
<div style="text-align: center; margin-top: 50px; color: #64748b; font-size: 0.8rem; border-top: 1px solid #1e293b; padding-top: 20px;">
    © 2025 Axiom Capital. <br>
    Past performance is not indicative of future results.
</div>
""", unsafe_allow_html=True)

import streamlit as st

# --- Advanced CSS for "Web-Like" Experience ---
st.markdown("""
<style>
    /* Hide Streamlit Elements */
    [data-testid="stSidebar"] {display: none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Reduce top padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Reset & Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@200;300;400;600;800&family=Playfair+Display:ital,wght@0,400;0,600;1,400&display=swap');
    
    html {
        scroll-behavior: smooth;
    }

    body {
        background-color: #020617; /* Slate-950 */
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #020617;
    }

    /* Navigation Bar */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px 40px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        background: rgba(2, 6, 23, 0.9); /* Slightly more opaque */
        backdrop-filter: blur(10px);
        position: sticky;
        top: 0;
        z-index: 1000;
    }
    
    .nav-logo {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 2.2rem;
        font-weight: 300;
        color: #f8fafc;
        letter-spacing: -0.02em;
        text-transform: none;
    }
    
    .nav-links {
        display: flex;
        gap: 30px;
        align-items: center;
    }
    
    .nav-item {
        color: #94a3b8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        text-decoration: none;
        transition: color 0.3s;
        cursor: pointer;
    }
    
    .nav-item:hover {
        color: #38bdf8;
    }
    
    .nav-btn {
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid #38bdf8;
        color: #38bdf8;
        padding: 8px 20px;
        border-radius: 4px;
        text-decoration: none;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.1em;
        transition: all 0.3s;
    }
    
    .nav-btn:hover {
        background: #38bdf8;
        color: #020617;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.3);
    }

    /* Hero Section */
    .hero-section {
        height: 90vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        background: radial-gradient(circle at 50% 50%, rgba(56, 189, 248, 0.1) 0%, rgba(2, 6, 23, 0) 60%);
        position: relative;
    }
    
    .hero-eyebrow {
        color: #38bdf8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        margin-bottom: 20px;
        animation: fadeIn 1s ease-out;
    }
    
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 5rem;
        font-weight: 400;
        line-height: 1.1;
        margin-bottom: 30px;
        background: linear-gradient(to right, #f8fafc, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: slideUp 1s ease-out;
    }
    
    .hero-desc {
        font-size: 1.2rem;
        color: #cbd5e1;
        max-width: 600px;
        line-height: 1.6;
        font-weight: 300;
        margin-bottom: 50px;
        animation: slideUp 1.2s ease-out;
    }
    
    /* Content Sections */
    .content-section {
        padding: 100px 40px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .section-header {
        text-align: center;
        margin-bottom: 60px;
    }
    
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        color: #f8fafc;
        margin-bottom: 20px;
    }
    
    .section-subtitle {
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 0.9rem;
    }
    
    /* Philosophy Grid */
    .philosophy-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 60px;
        max-width: 1200px;
        margin: 0 auto;
        align-items: center;
    }
    
    .philosophy-text {
        font-size: 1.1rem;
        color: #cbd5e1;
        line-height: 1.8;
    }
    
    .highlight-box {
        border-left: 4px solid #38bdf8;
        padding-left: 30px;
        margin-top: 30px;
        font-style: italic;
        color: #f1f5f9;
        font-family: 'Playfair Display', serif;
        font-size: 1.3rem;
    }

    /* Expertise Cards */
    .expertise-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 30px;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .expert-card {
        background: rgba(30, 41, 59, 0.3);
        padding: 40px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.3s;
    }
    
    .expert-card:hover {
        transform: translateY(-10px);
        background: rgba(30, 41, 59, 0.5);
        border-color: #38bdf8;
    }
    
    .expert-icon {
        font-size: 2.5rem;
        margin-bottom: 20px;
        color: #38bdf8;
    }
    
    .expert-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 15px;
    }

    /* Research Section */
    .research-container {
        max-width: 1000px;
        margin: 0 auto;
        text-align: center;
    }
    
    .stat-row {
        display: flex;
        justify-content: space-around;
        margin-top: 60px;
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-number {
        font-size: 4rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'Playfair Display', serif;
    }
    
    .stat-label {
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 0.8rem;
        margin-top: 10px;
    }

    /* Footer */
    .footer {
        padding: 60px 40px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        background: #020617;
    }

    /* Animations */
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    @keyframes slideUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    
    /* Button Override */
    div.stButton > button {
        background-color: transparent !important;
        border: 1px solid #f8fafc !important;
        color: #f8fafc !important;
        padding: 15px 40px !important;
        font-size: 1rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        border-radius: 0 !important;
        transition: all 0.3s !important;
    }
    
    div.stButton > button:hover {
        background-color: #f8fafc !important;
        color: #020617 !important;
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(255, 255, 255, 0.1);
    }

</style>
""", unsafe_allow_html=True)

# --- Navigation Bar (Functional Anchors + Portal Link) ---
st.markdown("""
<div class="navbar">
    <div class="nav-logo">Axiom <span style="font-weight: 700;">Capital</span></div>
    <div class="nav-links">
        <a href="#philosophy" class="nav-item">Philosophy</a>
        <a href="#expertise" class="nav-item">Expertise</a>
        <a href="landing?chat=true" class="nav-btn" target="_self">Portal</a>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown("""
<div class="hero-section">
    <div class="hero-eyebrow">Quantitative Asset Management</div>
    <div class="hero-title">Precision Engineering<br>for Your Financial Legacy</div>
    <div class="hero-desc">
        We harness the power of mathematics and advanced computing to construct 
        portfolios that withstand the test of time. No speculation. Just signal.
    </div>
</div>
""", unsafe_allow_html=True)

# --- CTA Button (Streamlit Interactive) ---
c1, c2, c3 = st.columns([1, 1, 1])
with c2:
    if st.button("Access Wealth Management Portal", use_container_width=True):
        st.session_state['auto_open_chat'] = True
        st.switch_page("pages/landing.py")

# --- Philosophy Section ---
st.markdown("<div id='philosophy'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="content-section" style="background: #0f172a;">
    <div class="section-header">
        <div class="section-subtitle">Our Core Beliefs</div>
        <div class="section-title">Mathematics Over Speculation</div>
    </div>
    <div class="philosophy-grid">
        <div>
            <p class="philosophy-text">
                At Axiom Capital, we reject the notion that consistent alpha can be generated through intuition or "gut feel". 
                The markets are a complex adaptive system, driven by millions of interacting agents.
            </p>
            <p class="philosophy-text" style="margin-top: 20px;">
                We believe that <b>Risk</b> is the only factor we can truly control. By rigorously defining our risk budget first, 
                we can engineer portfolios that maximize efficiency. We don't chase returns; we harvest risk premia systematically.
            </p>
        </div>
        <div>
            <div class="highlight-box">
                "We don't predict the future. We engineer portfolios that can thrive in any future."
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Expertise Section ---
st.markdown("<div id='expertise'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="content-section">
    <div class="section-header">
        <div class="section-subtitle">Capabilities</div>
        <div class="section-title">Expertise</div>
    </div>
    <div class="expertise-grid">
        <div class="expert-card">
            <div class="expert-icon">🛡️</div>
            <div class="expert-title">Risk Parity</div>
            <div class="value-text">
                Balancing risk contributions across asset classes to ensure true diversification, regardless of the economic regime.
            </div>
        </div>
        <div class="expert-card">
            <div class="expert-icon">⚡</div>
            <div class="expert-title">Convex Optimization</div>
            <div class="value-text">
                Solving complex utility maximization problems to find the precise asset weights that align with your risk tolerance.
            </div>
        </div>
        <div class="expert-card">
            <div class="expert-icon">🌱</div>
            <div class="expert-title">Sustainable Alpha</div>
            <div class="value-text">
                Integrating ESG constraints directly into the covariance matrix, proving that values and value can coexist.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div class="footer">
    <p>Geneva - Lausanne - Lonay - Vevey</p>
    <p style="margin-top: 20px;">© 2025 Axiom Capital.</p>
</div>
""", unsafe_allow_html=True)


import streamlit as st
import time

# --- Page Config & CSS ---
# Inherits layout="wide" from app.py

st.markdown("""
<style>
    [data-testid="stSidebar"] {display: none;}
    
    /* Reduce top padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Global Font */
    h1, h2, h3, p, label, .stRadio, .stSelectbox {
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

    /* --- Vertical Progress Bar (Stepper) --- */
    .stepper-container {
        position: fixed;
        top: 200px;
        left: 40px;
        border-left: 2px solid #334155;
        padding-left: 20px;
    }
    .step-item {
        margin-bottom: 30px;
        position: relative;
        color: #64748b;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .step-item::before {
        content: '';
        position: absolute;
        left: -26px;
        top: 5px;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #334155;
        border: 2px solid #1e293b;
        transition: all 0.3s ease;
    }
    .step-item.active {
        color: #f8fafc;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .step-item.active::before {
        background-color: #3b82f6; /* Default active color */
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
    }
    
    /* Specific Step Colors */
    .step-financial.active { color: #38bdf8; }
    .step-financial.active::before { background-color: #38bdf8; box-shadow: 0 0 10px rgba(56, 189, 248, 0.5); }
    
    .step-risk.active { color: #a855f7; }
    .step-risk.active::before { background-color: #a855f7; box-shadow: 0 0 10px rgba(168, 85, 247, 0.5); }
    
    .step-experience.active { color: #f97316; }
    .step-experience.active::before { background-color: #f97316; box-shadow: 0 0 10px rgba(249, 115, 22, 0.5); }

    /* --- Section Headers --- */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 1px solid #334155;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .header-financial { color: #38bdf8; border-color: #38bdf8; }
    .header-risk { color: #a855f7; border-color: #a855f7; }
    .header-experience { color: #f97316; border-color: #f97316; }

    /* --- Question Cards --- */
    .question-box {
        background-color: rgba(30, 41, 59, 0.4);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    .question-box:hover {
        transform: translateY(-2px);
    }
    
    /* Colored Borders per Section */
    .box-financial { border-left: 4px solid #38bdf8; }
    .box-financial:hover { box-shadow: 0 10px 30px -5px rgba(56, 189, 248, 0.1); border-color: #38bdf8; }
    
    .box-risk { border-left: 4px solid #a855f7; }
    .box-risk:hover { box-shadow: 0 10px 30px -5px rgba(168, 85, 247, 0.1); border-color: #a855f7; }
    
    .box-experience { border-left: 4px solid #f97316; }
    .box-experience:hover { box-shadow: 0 10px 30px -5px rgba(249, 115, 22, 0.1); border-color: #f97316; }

    .question-text {
        color: #e2e8f0;
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 15px;
    }

    /* Radio Button Styling */
    .stRadio > label { display: none; }
    div[role="radiogroup"] > label > div:first-of-type {
        background-color: #334155;
    }

    /* Submit Button (Friendly Green & Right Aligned) */
    [data-testid="stFormSubmitButton"] button {
        width: 100% !important;
        background-color: transparent !important;
        border: 2px solid #10b981 !important; /* Neon Green Border */
        color: #10b981 !important; /* Neon Green Text */
        padding: 12px 24px !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        border-radius: 12px !important; /* Slightly rounded, not pill */
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    [data-testid="stFormSubmitButton"] button:hover {
        background-color: #10b981 !important; /* Fill on Hover */
        color: #0f172a !important; /* Dark Text on Hover */
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.6) !important; /* Glow Effect */
        transform: translateY(-2px) !important;
    }
    [data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 20px -5px rgba(16, 185, 129, 0.4) !important;
        background: linear-gradient(90deg, #34d399 0%, #10b981 100%) !important;
    }
    
    /* Hero/Header */
    .axiom-header {
        text-align: center;
        margin-bottom: 50px;
        margin-top: -20px; /* Pull up closer to the button */
    }
    .axiom-title {
        font-size: 2.5rem;
        font-weight: 300;
        color: #f8fafc;
        margin-bottom: 5px;
    }
    .axiom-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 300;
    }

</style>
""", unsafe_allow_html=True)

# --- Navigation ---
if st.button("⬅️ Back to Home"):
    st.switch_page("pages/landing.py")

# --- Header ---
st.markdown("""
<div class="axiom-header">
    <div class="axiom-title">Axiom <span style="font-weight: 700; color: #94a3b8;">Capital</span></div>
    <div class="axiom-subtitle">Precision Profiling. Discover your true investor DNA.</div>
</div>
""", unsafe_allow_html=True)

# --- Layout: 2 Columns (Progress Bar | Form) ---
# Using columns to simulate the sidebar layout for the progress bar
col_progress, col_form = st.columns([1, 3], gap="large")

with col_progress:
    # Vertical Stepper (Visual Only)
    st.markdown("""
    <div style="position: fixed; top: 250px;">
        <div style="border-left: 2px solid #334155; padding-left: 20px;">
            <div class="step-item step-financial active">1. Financial Situation</div>
            <div class="step-item step-risk active">2. Risk Tolerance</div>
            <div class="step-item step-experience active">3. Experience</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_form:
    with st.form("investor_profile_form"):
        
        # --- Section 1: Financial Situation (Blue) ---
        st.markdown('<div class="section-header header-financial">1. Financial Situation</div>', unsafe_allow_html=True)
        
        # Q1
        st.markdown('<div class="question-box box-financial"><div class="question-text">What is your investment time horizon?</div>', unsafe_allow_html=True)
        q1 = st.radio("q1", ["Less than 1 year", "1-3 years", "3-7 years", "More than 7 years"], key="q1", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        # Q2
        st.markdown('<div class="question-box box-financial"><div class="question-text">What percentage of your income do you save annually?</div>', unsafe_allow_html=True)
        q2 = st.radio("q2", ["Less than 5%", "5-10%", "10-20%", "More than 20%"], key="q2", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        # Q3
        st.markdown('<div class="question-box box-financial"><div class="question-text">How would you describe your current financial situation?</div>', unsafe_allow_html=True)
        q3 = st.radio("q3", ["Unstable", "Stable", "Very stable"], key="q3", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)


        # --- Section 2: Risk Tolerance (Purple) ---
        st.markdown('<div class="section-header header-risk">2. Risk Tolerance</div>', unsafe_allow_html=True)

        # Q4
        st.markdown('<div class="question-box box-risk"><div class="question-text">If your portfolio dropped 20% in one month, what would you do?</div>', unsafe_allow_html=True)
        q4 = st.radio("q4", ["Sell everything", "Sell some", "Hold", "Buy more"], key="q4", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        # Q5
        st.markdown('<div class="question-box box-risk"><div class="question-text">What is your primary investment goal?</div>', unsafe_allow_html=True)
        q5 = st.radio("q5", ["Capital preservation", "Steady income", "Growth", "Aggressive growth"], key="q5", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        # Q6
        st.markdown('<div class="question-box box-risk"><div class="question-text">How do you react to market volatility?</div>', unsafe_allow_html=True)
        q6 = st.radio("q6", ["Very anxious", "Somewhat concerned", "Calm", "Excited"], key="q6", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        # Q7
        st.markdown('<div class="question-box box-risk"><div class="question-text">What maximum annual loss can you tolerate?</div>', unsafe_allow_html=True)
        q7 = st.radio("q7", ["Less than 5%", "5-10%", "10-20%", "More than 20%"], key="q7", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)


        # --- Section 3: Investment Experience (Orange) ---
        st.markdown('<div class="section-header header-experience">3. Investment Experience</div>', unsafe_allow_html=True)

        # Q8
        st.markdown('<div class="question-box box-experience"><div class="question-text">How many years of investment experience do you have?</div>', unsafe_allow_html=True)
        q8 = st.radio("q8", ["Less than 1 year", "1-3 years", "3-7 years", "More than 7 years"], key="q8", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        # Q9
        st.markdown('<div class="question-box box-experience"><div class="question-text">Which asset classes have you invested in?</div>', unsafe_allow_html=True)
        q9 = st.radio("q9", ["Only savings", "Bonds / ETFs", "Stocks", "Options / Derivatives"], key="q9", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        # Q10
        st.markdown('<div class="question-box box-experience"><div class="question-text">How often do you review your portfolio?</div>', unsafe_allow_html=True)
        q10 = st.radio("q10", ["Rarely", "Annually", "Quarterly", "Monthly or more"], key="q10", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        # Q11
        st.markdown('<div class="question-box box-experience"><div class="question-text">How would you rate your financial knowledge?</div>', unsafe_allow_html=True)
        q11 = st.radio("q11", ["Beginner", "Intermediate", "Advanced", "Expert"], key="q11", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("###")
        
        # Submit (Right Aligned)
        c1, c2 = st.columns([2, 1])
        with c2:
            submitted = st.form_submit_button("Analyze Profile")

# --- Scoring Logic ---
if submitted:
    score = 0
    
    # Q1 Horizon
    if q1 == "Less than 1 year": score += 5
    elif q1 == "1-3 years": score += 10
    elif q1 == "3-7 years": score += 15
    elif q1 == "More than 7 years": score += 20
    
    # Q2 Savings
    if q2 == "Less than 5%": score += 2
    elif q2 == "5-10%": score += 5
    elif q2 == "10-20%": score += 8
    elif q2 == "More than 20%": score += 10
    
    # Q3 Situation
    if q3 == "Unstable": score += 3
    elif q3 == "Stable": score += 6
    elif q3 == "Very stable": score += 10
    
    # Q4 Drop 20%
    if q4 == "Sell everything": score += 0
    elif q4 == "Sell some": score += 5
    elif q4 == "Hold": score += 10
    elif q4 == "Buy more": score += 15
    
    # Q5 Goal
    if q5 == "Capital preservation": score += 5
    elif q5 == "Steady income": score += 10
    elif q5 == "Growth": score += 15
    elif q5 == "Aggressive growth": score += 20
    
    # Q6 Volatility
    if q6 == "Very anxious": score += 2
    elif q6 == "Somewhat concerned": score += 5
    elif q6 == "Calm": score += 8
    elif q6 == "Excited": score += 10
    
    # Q7 Max Loss
    if q7 == "Less than 5%": score += 3
    elif q7 == "5-10%": score += 7
    elif q7 == "10-20%": score += 12
    elif q7 == "More than 20%": score += 15
    
    # Q8 Experience
    if q8 == "Less than 1 year": score += 2
    elif q8 == "1-3 years": score += 5
    elif q8 == "3-7 years": score += 8
    elif q8 == "More than 7 years": score += 10
    
    # Q9 Assets
    if q9 == "Only savings": score += 2
    elif q9 == "Bonds / ETFs": score += 5
    elif q9 == "Stocks": score += 8
    elif q9 == "Options / Derivatives": score += 10
    
    # Q10 Review
    if q10 == "Rarely": score += 2
    elif q10 == "Annually": score += 5
    elif q10 == "Quarterly": score += 8
    elif q10 == "Monthly or more": score += 10
    
    # Q11 Knowledge
    if q11 == "Beginner": score += 2
    elif q11 == "Intermediate": score += 5
    elif q11 == "Advanced": score += 8
    elif q11 == "Expert": score += 10
    
    # --- Redirection ---
    with st.spinner("Calculating optimal strategy..."):
        time.sleep(1.0) # UX pause
        
        if score <= 65:
            st.switch_page("pages/novice/landing_defensive.py")
        elif score <= 100:
            st.switch_page("pages/novice/landing_balanced.py")
        else:
            st.switch_page("pages/novice/landing_aggressive.py")

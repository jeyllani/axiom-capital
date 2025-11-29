import streamlit as st

st.set_page_config(page_title="QARM", layout="wide", initial_sidebar_state="collapsed")

# Define Pages
showcase_page = st.Page("pages/index.py", title="Axiom Capital", icon="🏛️", default=True)
home_page = st.Page("pages/landing.py", title="Wealth Portal", icon="🔐")
expert_home = st.Page("pages/experts/landing_expert.py", title="Expert Home", icon="🚀")
expert_utility = st.Page("pages/experts/expert_utility.py", title="Utility Maximization", icon="📊")
expert_risk = st.Page("pages/experts/expert_risk.py", title="Risk Architecture", icon="🛡️")
expert_frontier = st.Page("pages/experts/expert_frontier.py", title="Efficient Frontier", icon="📉")

# Navigation Setup
pg = st.navigation(
    {
        "Showcase": [showcase_page],
        "Main": [home_page],
        "Novice": [
            st.Page("pages/novice/questionnaire.py", title="Investor Profile", icon="🎯"),
            st.Page("pages/novice/landing_defensive.py", title="Defensive Strategies", icon="🛡️"),
            st.Page("pages/novice/landing_balanced.py", title="Balanced Strategies", icon="⚖️"),
            st.Page("pages/novice/landing_aggressive.py", title="Aggressive Strategies", icon="🚀"),
        ],
        "Intermediate": [
            st.Page("pages/products/landing_products.py", title="Product Catalog", icon="📦"),
        ],
        "Products": [
            st.Page("pages/products/defensive_1.py", title="Global Low Volatility Core", icon="🛡️"),
            st.Page("pages/products/defensive_2.py", title="Global Conservative Yield", icon="🛡️"),
            st.Page("pages/products/defensive_3.py", title="Global Moderate Growth", icon="🛡️"),
            st.Page("pages/products/balanced_1.py", title="Global Core Balanced", icon="⚖️"),
            st.Page("pages/products/balanced_2.py", title="Global Dynamic Growth", icon="⚖️"),
            st.Page("pages/products/balanced_3.py", title="Global Risk Parity", icon="⚖️"),
            st.Page("pages/products/aggressive_1.py", title="Global Dynamic Aggressive", icon="🚀"),
            st.Page("pages/products/aggressive_2.py", title="Global High Octane", icon="🚀"),
            st.Page("pages/products/aggressive_3.py", title="Global Max Return", icon="🚀"),
            st.Page("pages/products/esg_1.py", title="Global Sustainable Future", icon="🌱")
        ],
        "Expert Tools": [expert_home, expert_utility, expert_risk, expert_frontier]
    },
    position="hidden"  # Hides the default sidebar navigation
)

# Shared CSS for Margins (Applied globally)
st.markdown("""
<style>
    [data-testid="stSidebarNav"] {display: none;}
    [data-testid="stSidebar"] {display: none;}
    .block-container {padding-top: 4rem; padding-bottom: 0rem;}
</style>
""", unsafe_allow_html=True)

# --- Session State Init ---
if 'loader' not in st.session_state:
    st.session_state.loader = None

# Run Navigation
pg.run()

# --- Global Components (Chatbot) ---
# Only render chatbot if we are NOT on the Showcase page (index.py)
# We can check the title of the current page being run
if pg.title != "Axiom Capital":
    from src.components.chatbot import render_chatbot
    render_chatbot(page_title=pg.title)

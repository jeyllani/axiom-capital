import streamlit as st
from openai import OpenAI

def render_chatbot(page_title=None):
    """
    Renders a floating chatbot button using the native st.popover.
    This is the STABLE version.
    """
    
    # --- 1. Initialize State ---
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am the Axiom Capital AI Assistant. How can I help you navigate our quantitative strategies today?"}
        ]

    # --- 2. Initialize OpenAI Client ---
    if "OPENAI_API_KEY" not in st.secrets:
        # Fail silently or show a small toast to avoid breaking UI
        return

    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    # --- 3. System Prompt ---
    PAGE_DESCRIPTIONS = {
        "Axiom Capital": "The high-end showcase introduction page. Features a cinematic video background and a 'Enter Portal' button.",
        
        "Wealth Portal": """The main dashboard (Landing Page).
        Three main paths:
        1. Wealth Advisory (Novice): 'Guided Portfolio Construction'. For Private Clients. Uses a behavioral questionnaire to map risk tolerance.
        2. Strategic Funds (Intermediate): 'Curated Investment Catalog'. For Asset Allocators. Access to institutional-grade strategies (Defensive, Balanced, Aggressive).
        3. Quantitative Desk (Expert): 'Advanced Analytics Terminal'. For Professional Traders. Sandbox for custom risk aversion (Lambda), Efficient Frontier, and Stress Testing.
        Includes a Knowledge Center explaining Axiom's philosophy: 'Mathematics beats speculation'.""",
        
        "Expert Home": """The Gateway to Expert Tools.
        Options:
        1. Utility Maximization: Optimize portfolios using CRRA Utility functions.
        2. Risk Architecture: Deep dive into Value at Risk (VaR), CVaR, and Drawdown analysis.
        3. Efficient Frontier: Visual exploration of the Risk/Return trade-off (Markowitz).""",
        
        "Utility Maximization": "Expert Tool: Optimizes portfolios based on Utility Theory (CRRA). Users adjust Risk Aversion (Lambda) and see real-time weight adjustments.",
        "Risk Architecture": "Expert Tool: Advanced risk analysis. Features: Historical VaR, Parametric VaR, Monte Carlo CVaR, and Stress Testing scenarios.",
        "Efficient Frontier": "Expert Tool: Visualizes the Efficient Frontier. Allows users to pick points on the curve to see the corresponding portfolio composition.",
        
        "Product Catalog": """The Strategic Funds Catalog.
        Available Strategies:
        - Defensive Series: Global Low Volatility, Conservative Yield, Moderate Growth. (Focus: Capital Preservation).
        - Balanced Series: Global Core Balanced, Dynamic Growth, Risk Parity. (Focus: Risk-Adjusted Returns).
        - Aggressive Series: Global Dynamic Aggressive, High Octane, Max Return. (Focus: Capital Appreciation).
        - ESG Series: Global Sustainable Future. (Focus: Ethical/Sustainable constraints).""",
        
        "Investor Profile": "Novice Module: A psychometric questionnaire to determine the user's risk profile (Defensive, Balanced, or Aggressive).",
        
        # Novice Landings
        "Defensive Strategies": "Novice Module: Selection of low-risk strategies. Recommended for risk-averse investors focusing on preservation.",
        "Balanced Strategies": "Novice Module: Selection of balanced strategies. Recommended for investors seeking growth with moderate risk.",
        "Aggressive Strategies": "Novice Module: Selection of high-growth strategies. Recommended for risk-tolerant investors seeking maximum returns.",
        
        # Specific Products
        "Global Low Volatility Core": "Product Strategy: Defensive - Low Volatility Core. Minimizes variance to reduce drawdown risk.",
        "Global Conservative Yield": "Product Strategy: Defensive - Conservative Yield. Focuses on stable income generating assets.",
        "Global Moderate Growth": "Product Strategy: Defensive - Moderate Growth. A blend of safety and slight capital appreciation.",
        "Global Core Balanced": "Product Strategy: Balanced - Core Balanced. The standard 60/40 equivalent optimized for Sharpe.",
        "Global Dynamic Growth": "Product Strategy: Balanced - Dynamic Growth. Slightly higher equity exposure for long-term growth.",
        "Global Risk Parity": "Product Strategy: Balanced - Risk Parity. Allocates risk equally across asset classes (ERC).",
        "Global Dynamic Aggressive": "Product Strategy: Aggressive - Dynamic Aggressive. High equity exposure.",
        "Global High Octane": "Product Strategy: Aggressive - High Octane. Momentum-tilted high volatility strategy.",
        "Global Max Return": "Product Strategy: Aggressive - Max Return. Unconstrained optimization for maximum theoretical return.",
        "Global Sustainable Future": "Product Strategy: ESG - Sustainable Future. Excludes 'dirty' sectors (Energy, Utilities, Materials) while optimizing for Sharpe.",
    }

    SYSTEM_PROMPT = """
    You are the AI Investment Strategist for Axiom Capital.
    Tone: Institutional, trustworthy, precise.
    
    CONTEXT AWARENESS:
    1. Current Location: "{page_title}" - {page_desc}
    2. Memory: You have access to the results of the LAST optimization run (if any).
    
    INSTRUCTIONS:
    - If the user asks about the "current" strategy, refer to the "Last Optimization Run" data.
    - If the user is on a different page (e.g., Landing), explain that you still remember their last run.
    - Use the 'Current Location' description to understand what the user is seeing right now.
    """

    # --- 4. CSS for Floating Button (Safe Version) ---
    # We only target the popover container, which is standard.
    st.markdown("""
    <style>
        /* Position the popover container fixed at bottom right */
        [data-testid="stPopover"] {
            position: fixed;
            bottom: 30px;
            right: 30px;
            z-index: 9999;
        }
        
        /* Style the button inside */
        [data-testid="stPopover"] > button {
            width: 60px;
            height: 60px;
            border-radius: 30px;
            background: linear-gradient(135deg, #0f172a 0%, #334155 100%);
            border: 1px solid #38bdf8;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            color: #38bdf8;
            font-size: 24px;
        }
        
        [data-testid="stPopover"] > button:hover {
            transform: scale(1.1);
            background: linear-gradient(135deg, #1e293b 0%, #475569 100%);
        }
    </style>
    """, unsafe_allow_html=True)

    # --- 5. The Popover (Stable) ---
    with st.popover("💬", use_container_width=False):
        st.markdown("### 🤖 Axiom Assistant")
        st.markdown("---")
        
        # Display History
        messages_container = st.container(height=400)
        for msg in st.session_state.messages:
            with messages_container.chat_message(msg["role"]):
                st.write(msg["content"])
        
        # Input & Response
        if prompt := st.chat_input("Ask about strategies...", key="chat_input_widget"):
            # 1. Add User Message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with messages_container.chat_message("user"):
                st.write(prompt)
            
            # 2. Context Injection
            page_desc = PAGE_DESCRIPTIONS.get(str(page_title), "A section of the Axiom Capital application.")
            current_system_prompt = SYSTEM_PROMPT.replace("{page_title}", str(page_title)).replace("{page_desc}", page_desc)
            
            if 'latest_optimization_context' in st.session_state:
                ctx = st.session_state['latest_optimization_context']
                context_str = f"\n\n[MEMORY] Last Optimization Run Results: {ctx}"
                current_system_prompt += context_str
            else:
                current_system_prompt += "\n\n[MEMORY] No optimization run yet."
            
            # 3. API Call
            try:
                with messages_container.chat_message("assistant"):
                    stream = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "system", "content": current_system_prompt}] + st.session_state.messages,
                        stream=True,
                    )
                    response = st.write_stream(stream)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
            except Exception as e:
                st.error(f"Error: {e}")

import streamlit as st
from openai import OpenAI
import streamlit.components.v1 as components

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

    # --- 1.5. Check Query Params for Auto-Open ---
    # This handles the link from the navbar (landing?chat=true)
    try:
        query_params = st.query_params
        if query_params.get("chat") == "true":
            # st.toast("Chatbot: Auto-Open Triggered!", icon="🤖") # Debug
            st.session_state['auto_open_chat'] = True
            # Clear the param so it doesn't re-open on reload
            # Note: st.query_params is mutable in newer Streamlit versions
            query_params["chat"] = "false"
    except Exception as e:
        # st.toast(f"Chatbot Error: {e}", icon="⚠️") # Debug
        pass

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

    # --- 4. CSS for Floating Button (Fixed Width Version) ---
    st.markdown("""
    <style>
        /* 1. Cible le conteneur principal du popover */
        /* Note: Streamlit encapsule le popover dans une div avec data-testid="stPopover" */
        
        div[data-testid="stPopover"] {
            position: fixed !important;
            bottom: 30px !important;
            right: 30px !important;
            width: auto !important; /* EMPÊCHE DE PRENDRE TOUTE LA LARGEUR */
            height: auto !important;
            z-index: 99999 !important;
            background: transparent !important;
            border: none !important;
        }

        /* 2. Style du bouton rond (L'élément cliquable) */
        /* Target the button specifically inside the popover container */
        div[data-testid="stPopover"] button {
            width: 80px !important;
            height: 80px !important;
            min-width: 80px !important; /* Force min-width */
            min-height: 80px !important; /* Force min-height */
            border-radius: 50% !important; /* Rond parfait */
            background: linear-gradient(135deg, #0f172a 0%, #334155 100%) !important;
            border: 2px solid #38bdf8 !important; /* Bordure plus visible */
            box-shadow: 0 4px 15px rgba(56, 189, 248, 0.4) !important; /* Glow effect */
            color: #38bdf8 !important;
            font-size: 32px !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            transition: all 0.3s ease !important;
            margin: 0 !important;
        }
        
        /* HIDE THE CHEVRON (The 'v' arrow) */
        div[data-testid="stPopover"] button svg {
            display: none !important;
        }

        /* 3. Hover Effect */
        div[data-testid="stPopover"] > button:hover {
            transform: scale(1.1) !important;
            background: linear-gradient(135deg, #1e293b 0%, #475569 100%) !important;
            box-shadow: 0 6px 20px rgba(56, 189, 248, 0.6) !important;
            border-color: #7dd3fc !important;
        }

        /* 4. Gestion de la fenêtre ouverte (Le Chat lui-même) */
        div[data-testid="stPopoverBody"] {
            border: 1px solid rgba(56, 189, 248, 0.2) !important; /* Bordure subtile bleue */
            border-radius: 16px !important;
            box-shadow: 0 20px 50px rgba(0,0,0,0.6) !important;
            background: rgba(15, 23, 42, 0.95) !important; /* Slate 900 semi-transparent */
            backdrop-filter: blur(10px) !important; /* Effet verre */
            width: 400px !important; /* Largeur standard */
            max-height: 700px !important; /* Plus haut */
            padding: 0 !important; /* Remove default padding for edge-to-edge look */
            overflow-x: hidden !important; /* Hide horizontal scrollbar */
        }
        
        /* Petit fix pour éviter que le bouton "Submit" du formulaire prenne toute la largeur */
        /* High specificity to override landing_products.py column styles */
        div[data-testid="stPopoverBody"] [data-testid="column"] button,
        div[data-testid="stPopoverBody"] div[data-testid="stForm"] button {
            border-radius: 0 4px 4px 0 !important;
            background-color: #38bdf8 !important;
            background: #38bdf8 !important; /* Override gradient */
            background-image: none !important; /* Explicitly remove gradient image */
            color: #0f172a !important;
            border: none !important;
            border-top: none !important; /* Override column border */
            box-shadow: none !important;
            height: auto !important;
            padding: 10px !important;
            transform: none !important; /* Prevent hover transform */
            margin: 0 !important;
            min-height: 0 !important; /* Override min-height from cards */
        }
        
        /* Prevent Hover effects from landing page */
        div[data-testid="stPopoverBody"] [data-testid="column"] button:hover,
        div[data-testid="stPopoverBody"] div[data-testid="stForm"] button:hover {
            background-color: #7dd3fc !important;
            background: #7dd3fc !important;
            color: #0f172a !important;
            transform: none !important;
            box-shadow: none !important;
            border: none !important;
        }
        
        /* 5. Compact Text & Spacing */
        .stChatMessage p {
            font-size: 0.95rem !important;
            line-height: 1.5 !important;
        }
        .stChatMessage {
            padding-top: 0.5rem !important;
            padding-bottom: 0.5rem !important;
            background-color: transparent !important;
        }
        
        /* User Message Style */
        div[data-testid="chatAvatarIcon-user"] {
            background-color: #38bdf8 !important;
        }

        /* 6. Input Field Styling (Smaller Text) */
        div[data-testid="stPopoverBody"] input[type="text"] {
            font-size: 14px !important;
            height: 40px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # --- 5. The Popover (Stable) ---
    with st.popover("💬", use_container_width=False):
        # Header styled to match the box
        st.markdown(
            "<div style='background: rgba(30, 41, 59, 0.5); padding: 20px 15px 15px 15px; border-bottom: 1px solid rgba(255,255,255,0.1); margin: -1rem -1rem 5px -1rem;'>"
            "<h3 style='text-align: center; margin: 0; font-size: 1.2rem; color: #f8fafc;'>🤖 Axiom Assistant</h3>"
            "</div>", 
            unsafe_allow_html=True
        )
        
        # Display History
        messages_container = st.container(height=380)
        for msg in st.session_state.messages:
            with messages_container.chat_message(msg["role"]):
                st.write(msg["content"])
        
        # Input & Response (Using text_input + button for stability)
        # We use a form to allow 'Enter' to submit
        st.markdown("<div style='margin-top: -15px;'></div>", unsafe_allow_html=True) # Pull form up
        
        # Input & Response (Using text_input + button for stability)
        # We use a form to allow 'Enter' to submit
        with st.form(key="chat_form", clear_on_submit=True):
            cols = st.columns([0.85, 0.15])
            with cols[0]:
                user_input = st.text_input(
                    "Ask about strategies...", 
                    placeholder="Type your message...", 
                    label_visibility="collapsed",
                    key="chat_input_text"
                )
            with cols[1]:
                submit_button = st.form_submit_button("➤")
        
        if submit_button and user_input:
            # 1. Add User Message
            st.session_state.messages.append({"role": "user", "content": user_input})
            with messages_container.chat_message("user"):
                st.write(user_input)
            
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

    # --- 6. Auto-Open Logic (JS Injection) ---
    if st.session_state.get('auto_open_chat', False):
        # Inject JS to click the popover button
        # We use a slightly longer interval and a timeout to be safe
        components.html("""
            <script>
                console.log("Attempting to auto-open chatbot...");
                var attempts = 0;
                const checkExist = setInterval(function() {
                   attempts++;
                   // Target the button inside the popover container
                   const btn = window.parent.document.querySelector('div[data-testid="stPopover"] button');
                   if (btn) {
                      console.log("Chatbot button found, clicking...");
                      btn.click();
                      clearInterval(checkExist);
                   }
                   if (attempts > 50) { // 10 seconds
                       console.log("Chatbot button not found, giving up.");
                       clearInterval(checkExist);
                   }
                }, 200);
            </script>
        """, height=0, width=0)
        
        # Reset flag
        st.session_state['auto_open_chat'] = False

import streamlit as st
import os
from src.config import settings
from src.rag.chain import get_rag_chain

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SupportPearlz AI Assistant",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Custom CSS Styling
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .chat-subtitle {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Sidebar Setup
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("About SupportPearlz")
    st.write(
        "This AI assistant uses advanced RAG (Retrieval-Augmented Generation) "
        "backed by OpenAI and LangChain to answer customer support queries accurately."
    )
    st.divider()
    
    st.subheader("Support Channels:")
    st.markdown("• **Email:** support@supportpearlz.com")
    st.markdown("• **Phone:** 1-800-555-PEARL")
    st.divider()

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# -----------------------------------------------------------------------------
# 4. Initialize RAG Chain & Session State
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def initialize_chain():
    try:
        return get_rag_chain()
    except Exception as e:
        st.error(f"Failed to initialize RAG chain: {e}")
        return None

rag_chain = initialize_chain()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hello! Welcome to SupportPearlz. How can I help you with our products, shipping, or policies today?"
        }
    ]

# -----------------------------------------------------------------------------
# 5. Main UI & Chat Logic
# -----------------------------------------------------------------------------
st.markdown("<div class='chat-header'>💎 SupportPearlz Customer Support</div>", unsafe_allow_html=True)
st.markdown("<div class='chat-subtitle'>Your 24/7 AI assistant for policies, pricing, and product guidance.</div>", unsafe_allow_html=True)

if rag_chain is not None:
    st.success("Connected to SupportPearlz Knowledge Base successfully!", icon="✅")
else:
    st.warning("Knowledge base disconnected. Please verify that vector store index files are present.", icon="⚠️")

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input Box
if user_prompt := st.chat_input("Ask a question about SupportPearlz..."):
    # Append user prompt to state & screen
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Generate assistant response
    if rag_chain:
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                try:
                    # Invoke RAG chain
                    response = rag_chain.invoke(user_prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"An error occurred while generating a response: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        with st.chat_message("assistant"):
            st.error("RAG chain is unavailable. Please check backend configuration.")
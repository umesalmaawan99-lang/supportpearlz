import streamlit as st
import os

# Ensure vector store exists on Streamlit Cloud startup
if not os.path.exists("data/vector_store"):
    from src.ingestion.build_index import main as build_index
    build_index()

from src.config import settings
from src.rag.chain import get_rag_chain

# Page Configuration
st.set_page_config(
    page_title="SupportPearlz AI Assistant",
    page_icon="💎",
    layout="centered"
)

import streamlit as st
import os
from src.config import settings
from src.rag.chain import get_rag_chain

# Page Configuration
st.set_page_config(
    page_title="SupportPearlz AI Assistant",
    page_icon="💎",
    layout="centered"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; color: #2C3E50; font-weight: 700; margin-bottom: 0px; }
    .subtitle { font-size: 1.1rem; color: #7F8C8D; margin-bottom: 30px; }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<p class="main-title">💎 SupportPearlz Customer Support</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your 24/7 AI assistant for policies, pricing, and product guidance.</p>', unsafe_allow_html=True)

# Initialize RAG Chain with caching
@st.cache_resource
def load_chain():
    return get_rag_chain()

try:
    with st.spinner("Initializing SupportPearlz AI Knowledge Base..."):
        rag_chain = load_chain()
    st.success("Connected to SupportPearlz Knowledge Base successfully!", icon="✅")
except Exception as e:
    st.error(f"Failed to initialize RAG chain: {e}")
    st.stop()

# Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Welcome to SupportPearlz. How can I help you with our products, shipping, or policies today?"}
    ]

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Handling
if user_query := st.chat_input("Ask a question about SupportPearlz..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base & generating response..."):
            try:
                response = rag_chain.invoke(user_query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"An error occurred while processing your request: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar Info
with st.sidebar:
    st.markdown("### About SupportPearlz")
    st.write("This AI assistant uses advanced RAG (Retrieval-Augmented Generation) backed by OpenAI and LangChain to answer customer support queries accurately.")
    st.markdown("---")
    st.write("**Support Channels:**")
    st.write("- Email: support@supportpearlz.com")
    st.write("- Phone: 1-800-555-PEARL")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat history cleared. How can I help you today?"}
        ]
        st.rerun()
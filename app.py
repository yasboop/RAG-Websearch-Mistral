"""
Streamlit application for the Gromo RAG Chatbot.
"""
import os
import streamlit as st
from dotenv import load_dotenv

from src.utils import initialize_rag_system, format_chat_history, get_timestamp

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Gromo FAQ Chatbot",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stTextInput>div>div>input {
        background-color: white;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #e6f3ff;
        border-left: 5px solid #2b6cb0;
    }
    .chat-message.bot {
        background-color: #f0f0f0;
        border-left: 5px solid #718096;
    }
    .chat-message .message-content {
        display: flex;
        margin-top: 0.5rem;
    }
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .message {
        flex: 1;
    }
    .timestamp {
        font-size: 0.8rem;
        color: #718096;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    try:
        with st.spinner("Initializing chatbot..."):
            st.session_state.rag_chain = initialize_rag_system()
        st.success("Chatbot initialized successfully!")
    except Exception as e:
        st.error(f"Error initializing chatbot: {e}")
        st.stop()

# App header
st.title("💰 Gromo FAQ Chatbot")
st.markdown("""
This chatbot answers questions about Gromo's services using:
- Company FAQ database
- Real-time web search results
""")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This chatbot uses Retrieval-Augmented Generation (RAG) with the Qwen QwQ model to provide accurate answers about Gromo's services.
    
    It combines information from:
    - Gromo's FAQ database
    - Real-time web search results
    
    Ask questions about:
    - Gromo's services and features
    - Investment options
    - Fees and pricing
    - Account management
    - Security and privacy
    """)
    
    st.header("Settings")
    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.experimental_rerun()
    
    if st.button("Rebuild Vector Store"):
        with st.spinner("Rebuilding vector store..."):
            st.session_state.rag_chain = initialize_rag_system(force_rebuild=True)
        st.success("Vector store rebuilt successfully!")

# Display chat messages
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    timestamp = message.get("timestamp", "")
    
    if role == "user":
        with st.container():
            st.markdown(f"""
            <div class="chat-message user">
                <div class="message-content">
                    <img src="https://api.dicebear.com/7.x/personas/svg?seed=user" class="avatar" alt="User Avatar">
                    <div class="message">
                        <div>{content}</div>
                        <div class="timestamp">{timestamp}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    elif role == "assistant":
        with st.container():
            st.markdown(f"""
            <div class="chat-message bot">
                <div class="message-content">
                    <img src="https://api.dicebear.com/7.x/bottts/svg?seed=gromo" class="avatar" alt="Bot Avatar">
                    <div class="message">
                        <div>{content}</div>
                        <div class="timestamp">{timestamp}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Chat input
with st.container():
    user_input = st.text_input("Ask a question about Gromo:", key="user_input", placeholder="e.g., What investment options does Gromo offer?")
    
    if user_input:
        # Add user message to chat history
        timestamp = get_timestamp()
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": timestamp
        })
        
        # Display user message
        with st.container():
            st.markdown(f"""
            <div class="chat-message user">
                <div class="message-content">
                    <img src="https://api.dicebear.com/7.x/personas/svg?seed=user" class="avatar" alt="User Avatar">
                    <div class="message">
                        <div>{user_input}</div>
                        <div class="timestamp">{timestamp}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Generate response
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.rag_chain.invoke(user_input)
            except Exception as e:
                response = f"I'm sorry, I encountered an error while processing your request: {str(e)}"
        
        # Add assistant message to chat history
        timestamp = get_timestamp()
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": timestamp
        })
        
        # Display assistant message
        with st.container():
            st.markdown(f"""
            <div class="chat-message bot">
                <div class="message-content">
                    <img src="https://api.dicebear.com/7.x/bottts/svg?seed=gromo" class="avatar" alt="Bot Avatar">
                    <div class="message">
                        <div>{response}</div>
                        <div class="timestamp">{timestamp}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Clear input
        st.rerun()

# Footer
st.markdown("---")
st.markdown("Powered by Qwen QwQ and Langchain | © Gromo 2023") 
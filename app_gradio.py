"""
Gradio application for the Gromo RAG Chatbot.
"""
import os
import gradio as gr
from datetime import datetime

from src.utils import initialize_rag_system, get_timestamp

# Initialize the RAG system
print("Initializing RAG system...")
rag_chain = initialize_rag_system()
print("RAG system initialized successfully!")

# Define chat history
chat_history = []

def respond(message, chat_history):
    """
    Generate a response to the user's message.
    
    Args:
        message (str): User's message
        chat_history (list): Chat history
        
    Returns:
        tuple: Updated chat history with new message and response
    """
    # Get response from RAG chain
    bot_response = rag_chain.invoke(message)
    
    # Extract content if it's a structured response
    if hasattr(bot_response, 'content'):
        response_text = bot_response.content
    else:
        response_text = str(bot_response)
    
    # Format as messages for Gradio chatbot
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": response_text})
    
    return "", chat_history

def clear_history():
    return None

# Create Gradio interface
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("# 🤖 Gromo RAG Chatbot")
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(type="messages", height=600)
            
            with gr.Row():
                message = gr.Textbox(
                    placeholder="Ask me anything about Gromo...",
                    container=False,
                    scale=8
                )
                submit_btn = gr.Button("Submit", scale=1)
            
            clear_btn = gr.Button("Clear Conversation")
            
        with gr.Column(scale=1):
            gr.Markdown("### About Gromo Chatbot")
            
            with gr.Accordion("What can I ask?", open=True):
                gr.Markdown("""
                - Information about Gromo and its services
                - Details about specific financial products (Zest Money, Fi, etc.)
                - Commission rates and earning potential
                - How to become a Gromo partner
                - Product benefits and features
                """)
                
            with gr.Accordion("Information Sources", open=True):
                gr.Markdown("""
                This chatbot uses information from:
                - Gromo's FAQ database
                - Web search results for up-to-date information
                
                For the most accurate information, please refer to Gromo's official website or contact customer support.
                """)

    # Set up event handlers
    submit_btn.click(
        respond,
        inputs=[message, chatbot],
        outputs=[message, chatbot]
    )
    
    message.submit(
        respond,
        inputs=[message, chatbot],
        outputs=[message, chatbot]
    )
    
    clear_btn.click(
        clear_history,
        outputs=[chatbot]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True) 
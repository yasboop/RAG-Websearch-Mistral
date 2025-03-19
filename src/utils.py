"""
Utility functions for the Gromo RAG Chatbot.
"""
import os
import time
from typing import Dict, Any, Optional

from src.data_loader import prepare_faq_documents
from src.embeddings import create_vector_store, load_vector_store
from src.rag_chain import RAGChain


def initialize_rag_system(force_rebuild: bool = False) -> RAGChain:
    """
    Initialize the RAG system by loading or creating the vector store and RAG chain.
    
    Args:
        force_rebuild (bool, optional): Whether to force rebuilding the vector store. Defaults to False.
        
    Returns:
        RAGChain: The initialized RAG chain
    """
    # Check if vector store exists
    vector_store = None
    if not force_rebuild:
        vector_store = load_vector_store()
    
    # If vector store doesn't exist or force_rebuild is True, create it
    if vector_store is None or force_rebuild:
        print("Creating vector store...")
        # Prepare FAQ documents
        documents = prepare_faq_documents()
        
        # Create vector store
        vector_store = create_vector_store(documents)
    
    # Create RAG chain
    rag_chain = RAGChain(vector_store)
    
    return rag_chain


def format_chat_history(messages: list) -> str:
    """
    Format chat history for display.
    
    Args:
        messages (list): List of message dictionaries with 'role' and 'content' keys
        
    Returns:
        str: Formatted chat history
    """
    formatted = ""
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        
        if role == "user":
            formatted += f"👤 User: {content}\n\n"
        elif role == "assistant":
            formatted += f"🤖 GromoBot: {content}\n\n"
        elif role == "system":
            # Skip system messages in the display
            continue
    
    return formatted


def get_timestamp() -> str:
    """
    Get current timestamp in a readable format.
    
    Returns:
        str: Formatted timestamp
    """
    return time.strftime("%Y-%m-%d %H:%M:%S") 
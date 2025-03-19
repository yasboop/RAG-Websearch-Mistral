"""
Script to initialize the vector store with the FAQ data.
"""
import os
import sys
from src.utils import initialize_rag_system

if __name__ == "__main__":
    print("Initializing RAG system...")
    
    # Check if force rebuild flag is provided
    force_rebuild = "--force" in sys.argv
    
    if force_rebuild:
        print("Forcing rebuild of vector store...")
    
    # Initialize RAG system
    rag_chain = initialize_rag_system(force_rebuild=force_rebuild)
    
    print("RAG system initialized successfully!")
    print("You can now run the chatbot with: streamlit run app.py") 
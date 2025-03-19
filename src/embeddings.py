"""
Module for creating and managing embeddings for the RAG system.
"""
import os
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.config import EMBEDDING_MODEL_NAME, VECTOR_STORE_DIR


def get_embeddings_model():
    """
    Get the embeddings model for vector representations.
    
    Returns:
        HuggingFaceEmbeddings: The embeddings model
    """
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    return embeddings


def create_vector_store(documents: List[Document], persist: bool = True):
    """
    Create a vector store from documents.
    
    Args:
        documents (List[Document]): List of documents to add to the vector store
        persist (bool, optional): Whether to persist the vector store. Defaults to True.
        
    Returns:
        Chroma: The vector store
    """
    embeddings = get_embeddings_model()
    
    # Create vector store
    if persist:
        # Create directory if it doesn't exist
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        
        # Create persistent vector store
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=VECTOR_STORE_DIR
        )
        
        # Persist to disk
        vector_store.persist()
        print(f"Created and persisted vector store with {len(documents)} documents")
    else:
        # Create in-memory vector store
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings
        )
        print(f"Created in-memory vector store with {len(documents)} documents")
    
    return vector_store


def load_vector_store():
    """
    Load an existing vector store from disk.
    
    Returns:
        Chroma: The loaded vector store, or None if it doesn't exist
    """
    if not os.path.exists(VECTOR_STORE_DIR):
        print(f"Vector store directory {VECTOR_STORE_DIR} does not exist")
        return None
    
    embeddings = get_embeddings_model()
    
    try:
        vector_store = Chroma(
            persist_directory=VECTOR_STORE_DIR,
            embedding_function=embeddings
        )
        print(f"Loaded vector store from {VECTOR_STORE_DIR}")
        return vector_store
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None 
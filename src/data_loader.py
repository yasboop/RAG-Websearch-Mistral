"""
Module for loading and processing the Gromo FAQ dataset.
"""
import pandas as pd
import re
import html
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import FAQ_DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP


def clean_text(text: str) -> str:
    """
    Clean text by removing HTML tags and fixing special characters.
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Fix quotes and other special characters
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def load_faq_data() -> pd.DataFrame:
    """
    Load the FAQ dataset from CSV file and clean it.
    
    Returns:
        pd.DataFrame: DataFrame containing the cleaned FAQ data
    """
    try:
        # Read CSV file
        df = pd.read_csv(FAQ_DATA_PATH)
        
        # Clean question and answer columns
        df['question'] = df['question'].apply(clean_text)
        df['answer'] = df['answer'].apply(clean_text)
        
        # Remove rows with empty questions or answers
        df = df.dropna(subset=['question', 'answer'])
        df = df[(df['question'] != '') & (df['answer'] != '')]
        
        # Remove duplicate questions
        df = df.drop_duplicates(subset=['question'])
        
        print(f"Loaded and cleaned {len(df)} FAQ entries")
        return df
    except Exception as e:
        print(f"Error loading FAQ data: {e}")
        return pd.DataFrame(columns=["question", "answer"])


def convert_to_documents(df: pd.DataFrame) -> List[Document]:
    """
    Convert DataFrame to a list of Langchain Document objects.
    
    Args:
        df (pd.DataFrame): DataFrame containing the FAQ data
        
    Returns:
        List[Document]: List of Document objects
    """
    documents = []
    
    for _, row in df.iterrows():
        # Combine question and answer into a single text
        text = f"Question: {row['question']}\nAnswer: {row['answer']}"
        
        # Create metadata
        metadata = {
            "source": "gromo_faq",
            "question": row["question"]
        }
        
        # Create Document object
        doc = Document(page_content=text, metadata=metadata)
        documents.append(doc)
    
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks for better retrieval.
    
    Args:
        documents (List[Document]): List of Document objects
        
    Returns:
        List[Document]: List of chunked Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunked_documents = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunked_documents)} chunks")
    
    return chunked_documents


def prepare_faq_documents() -> List[Document]:
    """
    Prepare FAQ documents for vector store.
    
    Returns:
        List[Document]: List of processed Document objects
    """
    # Load FAQ data
    df = load_faq_data()
    
    # Convert to documents
    documents = convert_to_documents(df)
    
    # Split documents into chunks
    chunked_documents = split_documents(documents)
    
    return chunked_documents 
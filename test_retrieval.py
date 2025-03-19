"""
Test script to directly examine what's being retrieved from the vector store.
"""
import sys
from src.utils import initialize_rag_system

def test_retrieval(query):
    """
    Test the retrieval component of the RAG system.
    
    Args:
        query (str): The query to test retrieval for
    """
    # Initialize the RAG system
    print("Initializing RAG system...")
    rag_chain = initialize_rag_system()
    print("RAG system initialized successfully!")
    
    # Get documents from vector store
    print(f"\nQuery: {query}")
    faq_docs = rag_chain.retriever.invoke(query)
    
    # Print retrieved documents
    print(f"\nRetrieved {len(faq_docs)} documents from vector store:")
    for i, doc in enumerate(faq_docs):
        print(f"\n--- Document {i+1} ---")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("-" * 80)
    
    # Get web search results
    web_docs = rag_chain.web_search.search_web(query)
    
    # Print web search results
    print(f"\nRetrieved {len(web_docs)} documents from web search:")
    for i, doc in enumerate(web_docs):
        print(f"\n--- Web Document {i+1} ---")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")
        print("-" * 80)

if __name__ == "__main__":
    # Get query from command line arguments
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = "What is the payout for Personal and Business Loans?"
    
    test_retrieval(query) 
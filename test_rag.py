"""
Script to test the RAG system with a sample query.
"""
import sys
from src.utils import initialize_rag_system

# Sample queries to test
SAMPLE_QUERIES = [
    "What is GroMo?",
    "How can I earn through Gromo?",
    "When will I receive my payout?",
    "What are GroMo Points?",
    "How is the payout calculated?"
]

if __name__ == "__main__":
    print("Initializing RAG system...")
    
    # Initialize RAG system
    rag_chain = initialize_rag_system()
    
    print("\nRAG system initialized successfully!")
    
    # Get custom query from command line if provided
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\nTesting with custom query: {query}")
        
        # Generate response
        response = rag_chain.invoke(query)
        
        # Print response
        print("\nResponse:")
        print(response)
    else:
        # Test with sample queries
        print("\nTesting with sample queries:")
        
        for i, query in enumerate(SAMPLE_QUERIES, 1):
            print(f"\n{i}. Query: {query}")
            
            # Generate response
            response = rag_chain.invoke(query)
            
            # Print response
            print(f"Response: {response}")
            
            # Add separator
            print("-" * 80) 
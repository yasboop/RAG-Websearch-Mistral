"""
Script to test the Gromo RAG Chatbot with various questions.
"""
import sys
from src.utils import initialize_rag_system

# Sample questions about Gromo
SAMPLE_QUESTIONS = [
    "What is Gromo?",
    "What financial products can I sell through Gromo?",
    "How do I earn commissions on Gromo?",
    "What are the advantages of using Gromo?",
    "How do I sign up for Gromo?",
    "Is Gromo available in all of India?",
    "What kind of support does Gromo provide?",
    "How secure is Gromo for financial transactions?",
    "What are the commission rates for different products?",
    "How quickly can I withdraw my earnings from Gromo?"
]

def test_chatbot(custom_query=None):
    """
    Test the Gromo RAG Chatbot with either a custom query or sample questions.
    
    Args:
        custom_query: Optional custom query to test
    """
    print("Initializing RAG system...")
    rag_system = initialize_rag_system()
    
    if custom_query:
        # Test with custom query
        print(f"\n\n===== QUESTION: {custom_query} =====")
        response = rag_system.invoke(custom_query)
        
        # Extract the actual text content from the response object if needed
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
            
        print(f"\nRESPONSE: {response_text}")
    else:
        # Test with sample questions
        for i, question in enumerate(SAMPLE_QUESTIONS, 1):
            print(f"\n\n===== TEST {i}: {question} =====")
            response = rag_system.invoke(question)
            
            # Extract the actual text content from the response object if needed
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
                
            print(f"\nRESPONSE: {response_text}")

if __name__ == "__main__":
    # Check if a custom query was provided as command-line argument
    if len(sys.argv) > 1:
        custom_query = " ".join(sys.argv[1:])
        test_chatbot(custom_query)
    else:
        # Run with sample questions
        test_chatbot() 
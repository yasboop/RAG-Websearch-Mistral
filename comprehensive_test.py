"""
Comprehensive test script for the Gromo RAG Chatbot.
This script tests the chatbot with a variety of questions to evaluate its performance.
"""
import sys
import time
from src.utils import initialize_rag_system

def test_chatbot_comprehensively():
    """
    Test the chatbot with a variety of questions to evaluate its performance.
    """
    # Initialize the RAG system
    print("Initializing RAG system...")
    rag_chain = initialize_rag_system()
    print("RAG system initialized successfully!")
    
    # Define a variety of test questions across different categories
    test_questions = [
        # Questions about commissions/payouts
        "What is the payout for Personal and Business Loans?",
        "What are the commission rates for credit cards?",
        "How much commission do I get for mutual fund sales?",
        "What's the payout structure for Groww Demat account?",
        "How is commission calculated for Axis Bank accounts?",
        
        # Questions about products
        "What financial products can I sell through Gromo?",
        "Does Gromo offer insurance products?",
        "Can I sell mutual funds through Gromo?",
        "What are the best selling products on Gromo?",
        
        # Questions about eligibility
        "Who is eligible to become a Gromo partner?",
        "What are the requirements to sell financial products through Gromo?",
        "Do I need any certifications to sell insurance on Gromo?",
        
        # Questions about processes
        "How do I get my commission payout?",
        "When are commissions paid out?",
        "How do I track my sales on Gromo?",
        "What happens if a customer cancels their purchase?",
        
        # Random questions
        "What is Zest Money?",
        "How do I contact Gromo support?",
        "Is there a Gromo mobile app?"
    ]
    
    # Test each question and print the response
    for i, question in enumerate(test_questions):
        print(f"\n\n===== TEST QUESTION {i+1}/{len(test_questions)} =====")
        print(f"QUESTION: {question}")
        print("-" * 80)
        
        # Get response from the chatbot
        response = rag_chain.invoke(question)
        
        print("\nRESPONSE:")
        print(response)
        print("=" * 80)
        
        # Add a small delay to avoid overwhelming the API
        time.sleep(1)

if __name__ == "__main__":
    test_chatbot_comprehensively() 
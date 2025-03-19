"""
Test script for SERP API integration
"""
import os
from dotenv import load_dotenv
from src.web_search import WebSearchTool
from pprint import pprint

def test_serpapi():
    # Load environment variables
    load_dotenv()
    
    # Initialize WebSearchTool
    web_search = WebSearchTool()
    
    # Check if API key is set
    print(f"SERPAPI_API_KEY set: {bool(web_search.api_key)}")
    
    # Perform a test search
    query = "What financial products does Gromo offer?"
    print(f"Searching for: {query}")
    
    docs = web_search.search_web(query)
    
    # Print results
    print(f"Found {len(docs)} documents")
    for i, doc in enumerate(docs):
        print(f"\nDocument {i+1}:")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")

if __name__ == "__main__":
    test_serpapi() 
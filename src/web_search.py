"""
Module for integrating web search functionality using SERP API.
"""
from typing import List
import os
from serpapi import GoogleSearch
from langchain_core.documents import Document

from src.config import WEB_SEARCH_ENABLED, WEB_SEARCH_NUM_RESULTS


class WebSearchTool:
    """
    Tool for performing web searches and converting results to documents.
    """
    
    def __init__(self):
        """
        Initialize the web search tool.
        """
        self.enabled = WEB_SEARCH_ENABLED
        self.api_key = os.getenv("SERPAPI_API_KEY", "")
    
    def search_web(self, query: str) -> List[Document]:
        """
        Search the web for the given query and return results as documents.
        
        Args:
            query (str): The search query
            
        Returns:
            List[Document]: List of documents containing web search results
        """
        if not self.enabled:
            print("Web search is disabled")
            return []
        
        if not self.api_key:
            print("SERPAPI_API_KEY not set in environment variables")
            # Provide a fallback document when API key is not set
            fallback_doc = Document(
                page_content="Web search is currently unavailable. Please set SERPAPI_API_KEY in your .env file.",
                metadata={"source": "web_search_fallback"}
            )
            return [fallback_doc]
        
        try:
            # Append "Gromo" to the query to get more relevant results
            search_query = f"Gromo {query}"
            
            # Set up the search parameters
            params = {
                "q": search_query,
                "api_key": self.api_key,
                "num": WEB_SEARCH_NUM_RESULTS
            }
            
            # Perform the search
            search = GoogleSearch(params)
            results = search.get_dict()
            
            # Convert results to documents
            documents = []
            
            # Process organic results
            if "organic_results" in results:
                for result in results["organic_results"][:WEB_SEARCH_NUM_RESULTS]:
                    # Extract relevant information
                    title = result.get("title", "")
                    snippet = result.get("snippet", "")
                    link = result.get("link", "")
                    
                    # Create document content
                    content = f"Title: {title}\nSnippet: {snippet}\nSource: {link}"
                    
                    # Create metadata
                    metadata = {
                        "source": "web_search",
                        "title": title,
                        "link": link
                    }
                    
                    # Create document
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
            
            print(f"Found {len(documents)} web search results for query: {query}")
            return documents
        
        except Exception as e:
            print(f"Error during web search: {e}")
            # Provide a fallback document when web search fails
            fallback_doc = Document(
                page_content="Web search encountered an error. Relying on FAQ data only.",
                metadata={"source": "web_search_fallback"}
            )
            return [fallback_doc] 
"""
Module for implementing the RAG chain using Langchain.
"""
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import HuggingFacePipeline
from langchain_mistralai import ChatMistralAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, AutoModelForSeq2SeqLM

from src.config import (
    QWEN_MODEL_NAME, 
    MISTRAL_MODEL_NAME,
    MISTRAL_API_KEY,
    RAG_PROMPT_TEMPLATE, 
    SYSTEM_MESSAGE, 
    TOP_K_RETRIEVAL, 
    USE_QUANTIZED_MODEL,
    USE_MISTRAL_API,
    BNB_CONFIG
)
from src.web_search import WebSearchTool


def get_llm():
    """
    Initialize the language model for text generation.
    
    Returns:
        The language model for text generation
    """
    try:
        if USE_MISTRAL_API:
            # Use Mistral AI API
            print(f"Using Mistral AI API: {MISTRAL_MODEL_NAME}")
            
            # Initialize Mistral AI client
            llm = ChatMistralAI(
                model=MISTRAL_MODEL_NAME,
                mistral_api_key=MISTRAL_API_KEY,
                temperature=0.7,
                max_tokens=1024
            )
            
            return llm
        else:
            # Use a smaller model for testing
            model_name = "google/flan-t5-small"
            print(f"Using smaller model for testing: {model_name}")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # Create text generation pipeline
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512
            )
            
            # Wrap in Langchain pipeline
            llm = HuggingFacePipeline(pipeline=pipe)
            
            return llm
    
    except Exception as e:
        print(f"Error initializing language model: {e}")
        raise


def format_docs(docs: List[Document]) -> str:
    """
    Format a list of documents into a string for the context.
    
    Args:
        docs (List[Document]): List of documents
        
    Returns:
        str: Formatted context string with source tagging
    """
    # Separate FAQ documents from web search documents
    faq_docs = [doc for doc in docs if doc.metadata.get("source") == "gromo_faq"]
    web_docs = [doc for doc in docs if doc.metadata.get("source") == "web_search"]
    
    # Format FAQ documents
    faq_text = ""
    if faq_docs:
        faq_text = "=== GROMO OFFICIAL FAQ DATA ===\n" + "\n\n".join([
            f"FAQ ITEM: {doc.page_content}" for doc in faq_docs
        ]) + "\n\n"
    
    # Format web search documents
    web_text = ""
    if web_docs:
        web_text = "=== WEB SEARCH RESULTS ===\n" + "\n\n".join([
            f"WEB RESULT: {doc.page_content}" for doc in web_docs
        ])
    
    # Combine with FAQ documents first
    return faq_text + web_text


class RAGChain:
    """
    RAG chain that combines FAQ data and web search results.
    """
    
    def __init__(self, vector_store):
        """
        Initialize RAG chain.
        
        Args:
            vector_store: Vector store for document retrieval
        """
        self.vector_store = vector_store
        self.retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K_RETRIEVAL})
        self.web_search = WebSearchTool()  # Updated to correct class name
        self.use_web_search = True  # Flag to control web search usage
        
        if USE_MISTRAL_API:
            # Use direct LLM interface for Mistral API
            print(f"Using Mistral AI API: {MISTRAL_MODEL_NAME}")
        else:
            # For other models, create a standard LangChain RAG chain
            self.chain = create_rag_chain()
        
        print("RAG chain initialized successfully!")
    
    def _keyword_search(self, query: str) -> List[Document]:
        """
        Perform a keyword search on the documents for specific terms.
        
        Args:
            query (str): User query
            
        Returns:
            List[Document]: List of documents matching keywords
        """
        # Define keywords to look for (expanded list)
        primary_keywords = [
            "payout", "commission", "rate", "percentage", "earn", "payment", 
            "personal loan", "business loan", "gromo point", "credit card",
            "demat", "saving", "account", "mutual fund", "insurance", "fee",
            "eligibility", "requirement", "process", "track", "cancel", "contact",
            "support", "app", "mobile", "partner"
        ]
        
        # Extract product names from the query - expanded with more specific product names
        product_keywords = [
            "hdfc", "bajaj", "idfc", "axis", "groww", "paytm", "lic", "sbi",
            "icici", "kotak", "freecharge", "zest money", "zest", "lendingkart", "money", 
            "niyo", "fi", "federal bank", "credit", "loan", "emi", "card", "demat",
            "bank", "fintech", "invest", "indusind", "bob", "savings", "jupiter",
            "appreciate", "appreciate app", "angel one", "angel broking", "edelweiss"
        ]
        
        # Check if any keywords are in the query
        query_lower = query.lower()
        matched_primary = [k for k in primary_keywords if k in query_lower]
        matched_products = [k for k in product_keywords if k in query_lower]
        
        # Combine all matched keywords
        all_matched = matched_primary + matched_products
        
        if not all_matched:
            return []
        
        # Get all documents from the vector store
        all_docs = self.vector_store.similarity_search("", k=150)
        
        # Score documents based on keyword matches
        scored_docs = []
        for doc in all_docs:
            doc_lower = doc.page_content.lower()
            score = 0
            
            # Score based on primary keyword matches
            for k in matched_primary:
                if k in doc_lower:
                    score += 2
            
            # Score based on product keyword matches - higher weight for product matches
            for k in matched_products:
                if k in doc_lower:
                    score += 5
                    # Extra points for exact matches in question
                    try:
                        if "question:" in doc_lower and "answer:" in doc_lower:
                            question_part = doc_lower.split("question:")[1].split("answer:")[0].lower()
                            if k in question_part:
                                score += 10
                    except IndexError:
                        # Skip if the splitting fails
                        pass
            
            if score > 0:
                scored_docs.append((doc, score))
        
        # Sort by score and return top matches
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        matched_docs = [doc for doc, score in scored_docs[:15]]  # Return more documents for better coverage
        
        if matched_docs:
            print(f"Enhanced keyword search found {len(matched_docs)} documents matching: {', '.join(all_matched)}")
        
        return matched_docs
    
    def _product_specific_search(self, query: str) -> List[Document]:
        """
        Perform a search specifically for product-related queries.
        
        Args:
            query (str): User query
            
        Returns:
            List[Document]: List of documents specifically about products
        """
        try:
            # List of product names to look for
            products = [
                "zest money", "zest", "fi", "fi money", "federal bank", "axis bank",
                "hdfc", "bajaj", "idfc", "kotak", "groww", "paytm", "jupiter",
                "freecharge", "lic", "angel one", "demat", "appreciate"
            ]
            
            query_lower = query.lower()
            matched_products = [p for p in products if p in query_lower]
            
            if not matched_products:
                return []
            
            print(f"Searching specifically for product information about: {', '.join(matched_products)}")
            
            # For each matched product, do a dedicated search
            product_docs = []
            for product in matched_products:
                # Try several query variations
                variations = [
                    f"what is {product}",
                    f"{product} details",
                    f"{product} information",
                    f"{product} features"
                ]
                
                for variation in variations:
                    try:
                        results = self.vector_store.similarity_search(variation, k=5)
                        product_docs.extend(results)
                    except Exception as e:
                        print(f"Error searching for '{variation}': {e}")
                        continue
            
            # Remove duplicates
            seen_content = set()
            unique_docs = []
            
            for doc in product_docs:
                if doc.page_content not in seen_content:
                    unique_docs.append(doc)
                    seen_content.add(doc.page_content)
            
            return unique_docs[:10]
        except Exception as e:
            print(f"Error in product-specific search: {e}")
            return []
    
    def _query_expansion(self, query: str) -> List[str]:
        """
        Expand the query with related terms to improve retrieval.
        
        Args:
            query (str): Original query
            
        Returns:
            List[str]: List of expanded queries
        """
        expanded_queries = [query]
        
        # Add query variations based on common question patterns
        query_lower = query.lower()
        
        # Commission/payout related expansions
        if "commission" in query_lower or "payout" in query_lower or "earn" in query_lower:
            expanded_queries.append(f"{query} rate")
            expanded_queries.append(f"{query} percentage")
            expanded_queries.append(f"how much can I earn {query}")
        
        # Product related expansions
        if "product" in query_lower or "sell" in query_lower:
            expanded_queries.append("what financial products can I sell")
            expanded_queries.append("what products are available on gromo")
        
        # Eligibility related expansions
        if "eligible" in query_lower or "requirement" in query_lower:
            expanded_queries.append("who can become gromo partner")
            expanded_queries.append("eligibility requirements")
        
        return expanded_queries
    
    def _direct_question_lookup(self, query: str) -> List[Document]:
        """
        Directly look up specific questions in the FAQ.
        
        Args:
            query (str): User query
            
        Returns:
            List[Document]: List of documents with exact question matches
        """
        try:
            # List of common questions with their exact wording in the FAQ
            common_questions = [
                "What is the Payout for Personal and Business Loans?",
                "How is the payout calculated?",
                "What are GroMo Points?",
                "What is the value of 1 GroMo Point?",
                "GroMo Points",
                "GroMo Point value",
                "What is Zest Money?",
                "What is Fi?",
                "Is Fi a bank?",
                "What is Freecharge?",
                "What is Groww?",
                "What is Paytm Money?",
                "How do I track my sales?",
                "Where can I check my earnings?",
                "What are the commission rates for different products?"
            ]
            
            # Find the closest matching question
            query_lower = query.lower()
            matches = []
            
            # First try exact match
            for question in common_questions:
                if question.lower() in query_lower or query_lower in question.lower():
                    print(f"Found direct FAQ match: '{question}'")
                    # Get exact question from vector store
                    try:
                        results = self.vector_store.similarity_search(question, k=2)
                        matches.extend(results)
                    except Exception as e:
                        print(f"Error searching for '{question}': {e}")
                        continue
            
            # Special handling for common topics
            if "gromo point" in query_lower or ("point" in query_lower and "value" in query_lower):
                # Try both questions about GroMo Points
                print("Special handling for GroMo Points")
                try:
                    points_results = self.vector_store.similarity_search("What are GroMo Points?", k=2)
                    value_results = self.vector_store.similarity_search("What is the value of 1 GroMo Point?", k=2)
                    matches.extend(points_results)
                    matches.extend(value_results)
                except Exception as e:
                    print(f"Error in GroMo Points special handling: {e}")
            
            # Special handling for specific products
            if "zest" in query_lower or "zest money" in query_lower:
                print("Special handling for Zest Money")
                try:
                    zest_results = self.vector_store.similarity_search("Zest Money", k=5)
                    matches.extend(zest_results)
                except Exception as e:
                    print(f"Error in Zest Money special handling: {e}")
                    
            if "fi" in query_lower and len(query_lower) < 10:  # Avoid matching "financial", "find", etc.
                print("Special handling for Fi")
                try:
                    fi_results = self.vector_store.similarity_search("Fi", k=5)
                    fi_bank_results = self.vector_store.similarity_search("Fi bank", k=5)
                    matches.extend(fi_results)
                    matches.extend(fi_bank_results)
                except Exception as e:
                    print(f"Error in Fi special handling: {e}")
            
            return matches
        except Exception as e:
            print(f"Error in direct question lookup: {e}")
            return []
    
    def _hybrid_search(self, query: str, top_k: int = 6) -> List[Document]:
        """
        Performs a hybrid search using both vector similarity and keyword matching.
        
        Args:
            query: The search query
            top_k: Number of results to return (reduced from 10 to 6 for more focused results)
            
        Returns:
            List of documents from the search
        """
        # Get results from vector store (dense retrieval)
        vector_results = self.vector_store.similarity_search(query, k=top_k)
        
        # Get results from keyword search (sparse retrieval)
        keyword_results = self._keyword_search(query)
        
        # Try product specific search if applicable
        product_results = self._product_specific_search(query)
        
        # Try direct question lookup for exact matches
        direct_results = self._direct_question_lookup(query)
        
        # Deduplicate and rank results
        all_results = []
        
        # Prioritize exact matches from direct lookup
        if direct_results:
            all_results.extend(direct_results[:2])  # Only take top 2 direct matches
        
        # Next prioritize product-specific results
        if product_results:
            # Filter to avoid duplicates
            new_products = [doc for doc in product_results[:3] 
                          if doc.page_content not in [d.page_content for d in all_results]]
            all_results.extend(new_products)
        
        # Add vector results (dense retrieval)
        for doc in vector_results:
            if doc.page_content not in [d.page_content for d in all_results]:
                all_results.append(doc)
                if len(all_results) >= top_k:
                    break
        
        # If we still need more, add keyword results
        if len(all_results) < top_k:
            for doc in keyword_results:
                if doc.page_content not in [d.page_content for d in all_results]:
                    all_results.append(doc)
                    if len(all_results) >= top_k:
                        break
        
        return all_results[:top_k]
    
    def _get_context(self, query: str) -> str:
        """
        Retrieves context for the query using multiple retrieval methods
        and formats it for the LLM.
        
        Args:
            query: The user query
            
        Returns:
            Formatted context string
        """
        # Retrieve relevant documents with hybrid search
        docs = self._hybrid_search(query, top_k=6)  # Reduced from 10 to 6 for more focused results
        
        # Get web search results only if needed
        web_results = []
        if self.use_web_search and len(docs) < 4:  # Only use web search if we have few relevant docs
            try:
                web_results = self.web_search.search_web(query)
                # Limit to top 2 web results to avoid overwhelming the context
                web_results = web_results[:2]
            except Exception as e:
                print(f"Web search failed: {e}")
        
        # Combine and format the context
        context = ""
        if docs:
            faq_section = format_docs(docs)
            context += f"Retrieved {len(docs)} documents from FAQ:\n{faq_section}\n"
        
        if web_results:
            web_section = format_docs(web_results)
            context += f"Found {len(web_results)} web search results for query: {query}\n{web_section}\n"
        
        return context
    
    def invoke(self, query: str) -> str:
        """
        Process a query and return a response.
        
        Args:
            query: User query
            
        Returns:
            Response from the LLM
        """
        try:
            print(f"\n\n===== PROCESSING QUERY: {query} =====")
            
            # Get context for the query
            print("Starting retrieval...")
            context = self._get_context(query)
            print(f"Retrieved context length: {len(context)} characters")
            
            # Create prompt with context and query
            prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=query)
            print(f"Total prompt length: {len(prompt)} characters")
            
            # Generate response
            print("Generating LLM response...")
            llm = get_llm()
            response = llm.invoke(prompt)
            print("LLM response generated")
            
            # Return the response (either as a string or with its content attribute)
            return response
        
        except Exception as e:
            print(f"Error in RAG chain: {e}")
            error_message = (
                "I apologize, but I encountered an error while processing your question. "
                "This might be due to the complexity of your query or technical limitations. "
                "Could you try rephrasing your question? Or if you're asking about "
                "specific product details, you may want to contact Gromo's customer support "
                "for the most accurate and up-to-date information."
            )
            return error_message 
"""
Configuration settings for the Gromo RAG Chatbot.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Model settings
USE_QUANTIZED_MODEL = False  # Set to True to use quantized model for AWS deployment
USE_MISTRAL_API = True  # Set to True to use Mistral AI API
QWEN_MODEL_NAME = "Qwen/Qwen2-7B-Instruct"  # Qwen model
MISTRAL_MODEL_NAME = "mistral-large-latest"  # Mistral AI model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model for vector store

# Quantization settings
QUANTIZATION_TYPE = "4bit"  # Options: "4bit", "8bit"
BNB_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "float16"
}

# RAG settings
CHUNK_SIZE = 512  # Size of text chunks for splitting documents
CHUNK_OVERLAP = 50  # Overlap between chunks
TOP_K_RETRIEVAL = 10  # Number of documents to retrieve from vector store

# Web search settings
WEB_SEARCH_ENABLED = True  # Enable/disable web search
WEB_SEARCH_NUM_RESULTS = 3  # Number of web search results to retrieve

# Vector store settings
VECTOR_STORE_DIR = "vector_store"  # Directory to store vector database

# Data settings
FAQ_DATA_PATH = "/Users/anandkumar/Downloads/gromo_RAG+websearch/gromo-faq-v1-0.csv"  # Path to FAQ dataset

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # OpenAI API key (optional)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")  # Mistral AI API key
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")  # SERP API key for web search

# AWS deployment settings
AWS_REGION = "us-east-1"  # AWS region for deployment
AWS_INSTANCE_TYPE = "g4dn.xlarge"  # AWS EC2 instance type for deployment
AWS_LAMBDA_MEMORY = 10240  # Memory allocation for AWS Lambda (MB) - maximum allowed
AWS_LAMBDA_TIMEOUT = 60  # Timeout for AWS Lambda (seconds)

# Prompt templates
RAG_PROMPT_TEMPLATE = """
You are GromoBot, the official AI assistant for Gromo, a financial technology platform that helps users sell financial products and earn commissions.

Your responses should be:
1. Helpful and informative, providing accurate information about Gromo's services and products
2. Friendly and conversational, but professional
3. Concise but comprehensive
4. Based on information from BOTH the retrieved FAQ documents AND web search results

Context information from Gromo's FAQ and web search:
{context}

When answering questions:
- Synthesize information from both FAQ data and web search results into a comprehensive answer
- For Gromo-specific information (commission rates, policies, products), prioritize the FAQ data
- Use web search results to add relevant details, especially for general information about financial products

When discussing specific financial products (like Zest Money, Fi, Groww, etc.):
- Thoroughly explain what the product is and its key features
- Detail how it relates to Gromo and how partners can earn from it
- Include specific benefits, eligibility criteria, and requirements when available

User query: {question}

RESPONSE:
"""

# System message for the LLM
SYSTEM_MESSAGE = """
You are GromoBot, the official AI assistant for Gromo, a financial technology platform that helps users sell financial products and earn commissions.

Your responses should be:
1. Helpful and informative, providing accurate information about Gromo's services and products
2. Friendly and conversational, but professional
3. Concise but comprehensive
4. Based on information from BOTH the retrieved FAQ documents AND web search results

When answering questions:
- Treat the FAQ data as the primary authoritative source for Gromo-specific information
- Use web search results to supplement and enrich your answers, especially for:
  * Current or time-sensitive information
  * Broader context about financial products or services
  * Details not covered in the FAQ
- Clearly synthesize both sources into a coherent, comprehensive answer

When discussing specific financial products (like Zest Money, Fi, Groww, etc.):
- Provide detailed information about what these products are
- Explain their relationship with Gromo (e.g., products that can be sold through Gromo)
- Include details about features, benefits, or requirements if available
- For questions about banks or fintech companies, explain what they are and how they relate to Gromo

If the FAQ and web search provide conflicting information, prioritize the FAQ data but acknowledge differences if they're significant.

If you don't know the answer or if the information isn't in the provided context, acknowledge that and suggest the user contact Gromo's customer support for more specific information.
""" 
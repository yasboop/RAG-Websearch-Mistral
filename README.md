# 🤖 Gromo FAQ RAG Chatbot with Web Search

![Gromo RAG Chatbot Screenshot](https://i.postimg.cc/xdWzXhkR/temp-Imageb95r-MZ.avif)

A powerful Retrieval-Augmented Generation (RAG) chatbot for Gromo that provides intelligent responses by combining:
1. 📚 **Company FAQ dataset** - For accurate company-specific information
2. 🔎 **Real-time web search results** - Using SERP API for up-to-date information

## ✨ Features

- **🧠 Advanced LLM Support**:
  - [Mistral AI API](https://mistral.ai/) (recommended for best performance)
  - Qwen QwQ model (local deployment)
  - Configurable for various models based on your needs
  
- **📊 Enhanced Context Understanding**:
  - Implements RAG architecture using Langchain
  - Vector database for efficient similarity search
  - Contextual response generation with relevant information retrieval
  
- **🌐 Real-time Web Search Integration**:
  - SERP API integration for current information
  - Blends FAQ knowledge with web search results
  - Provides citations and sources for information
  
- **🖥️ Multiple Interface Options**:
  - Gradio UI with modern interface (recommended)
  - Streamlit UI alternative
  - FastAPI endpoints for application integration
  
- **🛠️ Developer-Friendly Design**:
  - Modular code structure
  - Extensive documentation
  - Easy customization options

## 🗂️ Project Structure

```
gromo_rag/
├── data/                  # Directory for FAQ dataset
├── src/                   # Source code
│   ├── __init__.py
│   ├── config.py          # Configuration settings
│   ├── data_loader.py     # Load and process FAQ data
│   ├── embeddings.py      # Vector embeddings for RAG
│   ├── rag_chain.py       # RAG implementation with Langchain
│   ├── web_search.py      # Web search integration with SERP API
│   └── utils.py           # Utility functions
├── app_gradio.py          # Gradio application (recommended)
├── app.py                 # Streamlit application
├── app_fastapi.py         # FastAPI application
├── test_chatbot.py        # Script to test the chatbot
├── init_vector_store.py   # Script to initialize vector store
├── .env                   # Environment variables (not tracked by git)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yasboop/RAG-Websearch-Mistral.git
cd RAG-Websearch-Mistral
```

### 2. Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables:

Create a `.env` file with your API keys:

```
# Mistral AI API key
MISTRAL_API_KEY=your_mistral_api_key_here

# SERP API key for Google search
SERPAPI_API_KEY=your_serpapi_key_here
```

### 4. Initialize the vector store:

```bash
python init_vector_store.py
```

### 5. Run the application:

```bash
# Using Gradio (recommended)
python app_gradio.py

# Using Streamlit
streamlit run app.py

# Using FastAPI
python app_fastapi.py
```

## 🧪 Testing

Test the chatbot functionality with sample questions:

```bash
python test_chatbot.py
```

Or test with a specific question:

```bash
python test_chatbot.py "What financial products does Gromo offer?"
```

## ⚙️ Customization

Customize the chatbot by modifying the configuration in `src/config.py`:

- **Model Settings**:
  - Use Mistral AI API: `USE_MISTRAL_API = True`
  - Use local Qwen model: `USE_MISTRAL_API = False` and `USE_QUANTIZED_MODEL = False`
  - Use quantized model for AWS deployment: `USE_QUANTIZED_MODEL = True`
  
- **RAG Parameters**:
  - Adjust context window size
  - Modify similarity search parameters
  - Fine-tune retrieval strategy
  
- **Web Search Options**:
  - Enable/disable web search
  - Change search result count
  - Adjust search relevance parameters

## 🔌 API Usage

The chatbot can be integrated into other applications through the FastAPI interface:

1. Start the API server:
   ```bash
   python app_fastapi.py
   ```

2. Send requests to the API:
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "What is Gromo?"}'
   ```

## 🔑 Getting a SERP API Key

To use the web search functionality:

1. Visit [https://serpapi.com/](https://serpapi.com/) and sign up for an account
2. Navigate to your dashboard to find your API key
3. Add the API key to your `.env` file as `SERPAPI_API_KEY`

## 🌟 Advanced Usage

### AWS Deployment

This project includes scripts for AWS EC2 deployment. See `AWS_DEPLOYMENT.md` for detailed instructions.

### Custom Knowledge Base

To use a different knowledge base:
1. Replace the CSV file in the `data/` directory
2. Update the data loading logic in `src/data_loader.py` if necessary
3. Reinitialize the vector store with `python init_vector_store.py`

## 📄 License

MIT 
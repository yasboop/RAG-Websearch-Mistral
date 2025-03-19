# Gromo FAQ RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for Gromo that answers questions based on:
1. Company FAQ dataset
2. Real-time web search results (using SERP API)

## Features

- Supports multiple LLM options:
  - Mistral AI API (recommended for best performance)
  - Qwen QwQ model (local deployment)
  - Smaller models for testing
- Implements RAG using Langchain for context-aware responses
- Integrates web search via SERP API to provide up-to-date information
- Multiple interface options:
  - Gradio UI (recommended for ease of use)
  - Streamlit UI
  - FastAPI endpoints
- Processes and cleans the Gromo FAQ dataset

## Project Structure

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

## Quick Start

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your API keys:
   ```
   # Mistral AI API key
   MISTRAL_API_KEY=your_mistral_api_key_here

   # SERP API key for Google search
   SERPAPI_API_KEY=your_serpapi_key_here
   ```

3. Initialize the vector store:
   ```
   python init_vector_store.py
   ```

4. Run the application with your preferred interface:
   ```
   # Using Gradio (recommended)
   python app_gradio.py

   # Using Streamlit
   streamlit run app.py

   # Using FastAPI
   python app_fastapi.py
   ```

## Testing

Test the chatbot with sample questions:

```
python test_chatbot.py
```

Or test with a specific question:

```
python test_chatbot.py "What financial products does Gromo offer?"
```

## Customization

You can customize the chatbot by modifying the configuration in `src/config.py`:

- Change the model settings:
  - Use Mistral AI API by setting `USE_MISTRAL_API = True`
  - Use local Qwen model by setting `USE_MISTRAL_API = False` and `USE_QUANTIZED_MODEL = False`
  - Use quantized model for AWS deployment by setting `USE_QUANTIZED_MODEL = True`
- Adjust the RAG parameters
- Enable/disable web search
- Modify the prompt templates

## API Usage

You can also use the chatbot through the FastAPI interface:

1. Start the API server:
   ```
   python app_fastapi.py
   ```

2. Send requests to the API:
   ```
   curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"query": "What is Gromo?"}'
   ```

## Getting a SERP API Key

To use the web search functionality, you need to get a SERP API key:

1. Visit [https://serpapi.com/](https://serpapi.com/) and sign up for an account
2. Navigate to your dashboard to find your API key
3. Add the API key to your `.env` file as `SERPAPI_API_KEY`

## License

MIT 
# Gromo RAG Chatbot - Installation Guide for Beginners

This guide will help you set up and run the Gromo RAG Chatbot on your computer. Follow these steps carefully.

## Prerequisites

Before you begin, make sure you have:

1. Python 3.8 or higher installed on your computer
2. Basic knowledge of using the terminal/command prompt

## Step 1: Set Up Your Environment

1. Open your terminal/command prompt
2. Navigate to the project directory:
   ```
   cd /Users/anandkumar/Downloads/gromo_RAG+websearch
   ```

## Step 2: Create a Virtual Environment (Optional but Recommended)

A virtual environment helps keep your project dependencies separate from other Python projects.

1. Create a virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```
   - On Windows:
     ```
     venv\Scripts\activate
     ```

## Step 3: Install Dependencies

Install all the required packages:

```
pip install -r requirements.txt
```

This might take a few minutes as it installs all the necessary libraries.

## Step 4: Set Up Environment Variables

1. Create a `.env` file in the project directory:
   ```
   cp .env.example .env
   ```

2. Edit the `.env` file and add your OpenAI API key (optional, used as fallback):
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Step 5: Initialize the Vector Store

Before running the chatbot, you need to initialize the vector store with the FAQ data:

```
python init_vector_store.py
```

This will process the FAQ data and create a vector store for efficient retrieval.

## Step 6: Test the RAG System (Optional)

You can test if the RAG system is working correctly:

```
python test_rag.py
```

This will run a few sample queries to check if the system is responding correctly.

## Step 7: Run the Chatbot

Now you can run the chatbot:

```
streamlit run app.py
```

This will start the Streamlit server and open the chatbot interface in your web browser.

## Troubleshooting

If you encounter any issues:

1. **Error installing dependencies**: Make sure you have the latest version of pip:
   ```
   pip install --upgrade pip
   ```

2. **Model download issues**: The first time you run the chatbot, it will download the Qwen QwQ model, which might take some time depending on your internet connection.

3. **Memory issues**: If you encounter memory errors, you might need to reduce the model size or run on a machine with more RAM.

4. **Vector store errors**: If you encounter errors with the vector store, try rebuilding it:
   ```
   python init_vector_store.py --force
   ```

## Next Steps

Once you're comfortable with the basic setup, you can:

1. Customize the chatbot by modifying the configuration in `src/config.py`
2. Add more FAQ data to improve the chatbot's knowledge
3. Experiment with different models and parameters

## Getting Help

If you need further assistance, you can:

1. Check the documentation in the README.md file
2. Contact Gromo support for specific questions about the FAQ data
3. Look up error messages online for general Python or library-specific issues 
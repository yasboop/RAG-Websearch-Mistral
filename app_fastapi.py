"""
FastAPI application for the Gromo RAG Chatbot.
"""
import os
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from src.utils import initialize_rag_system

# Load environment variables
load_dotenv()

# Initialize the RAG system
rag_chain = initialize_rag_system()

# Create FastAPI app
app = FastAPI(
    title="Gromo FAQ Chatbot API",
    description="API for the Gromo FAQ Chatbot using RAG",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Define request and response models
class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str

@app.get("/")
async def root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "Gromo FAQ Chatbot API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint to get a response from the chatbot.
    
    Args:
        request (ChatRequest): The chat request containing the query
        
    Returns:
        ChatResponse: The chat response containing the answer
    """
    try:
        # Get query from request
        query = request.query
        
        # Generate response using RAG chain
        response = rag_chain.invoke(query)
        
        # Create conversation ID if not provided
        conversation_id = request.conversation_id or "new_conversation"
        
        # Return response
        return ChatResponse(
            response=response,
            conversation_id=conversation_id
        )
    
    except Exception as e:
        # Log error
        print(f"Error processing chat request: {e}")
        
        # Raise HTTP exception
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI app with uvicorn
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=8000, reload=True) 
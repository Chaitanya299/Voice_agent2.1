from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

from app.config import CHROMA_DB_PATH, PHI2_MODEL_PATH, LLAMA3B_MODEL_PATH

# Load the LLM model (using LlamaCpp for GGUF)
# Ensure you have either phi-2.gguf or llama-3b.gguf in the models/ directory
try:
    # Choose the model you want to use
    model_path = LLAMA3B_MODEL_PATH
    
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=2048, # Context window size
        n_gpu_layers=-1, # Offload all layers to GPU if available
        verbose=False, # Suppress verbose output
    )
except Exception as e:
    print(f"Error loading LLM model for RAG: {e}")
    llm = None

# Load the embeddings model
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except Exception as e:
    print(f"Error loading embeddings model for RAG: {e}")
    embeddings = None

# Load the Chroma vector store
try:
    vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
except Exception as e:
    print(f"Error loading Chroma vector store: {e}")
    vectorstore = None
    retriever = None

def get_rag_response(query: str) -> str:
    """
    Generates a response using a simple RAG flow:
    1. Retrieve relevant documents from Chroma
    2. Inject context into the prompt
    3. Ask the LLM to answer
    """
    if llm is None or retriever is None:
        return "RAG system not initialized. Cannot generate context-aware reply."
    try:
        docs = retriever.invoke(query)

        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
You are a restaurant voice assistant.

Answer ONLY the user's question.
Do NOT include unrelated information.
Be concise and specific.
If the answer is not present, say you don't know.

Context:
{context}

User question:
{query}

Answer (1–2 sentences max):
"""


        response = llm.invoke(prompt)
        return response
    except Exception as e:
        return f"Error generating RAG response: {e}"

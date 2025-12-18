# -*- coding: utf-8 -*-
import os
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import KNOWLEDGE_BASE_PATH, CHROMA_DB_PATH

def build_vector_index():
    """
    
    Builds a Chroma vector index using a local SentenceTransformer model.
    """
    print("Starting to build Chroma vector index with a local model...")
    
    try:
        with open(KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as f:
            knowledge_base_text = f.read()
        print(f"Successfully loaded knowledge base from: {KNOWLEDGE_BASE_PATH}")
    except FileNotFoundError:
        print(f"Error: Knowledge base file not found at {KNOWLEDGE_BASE_PATH}")
        return

    text_splitter = CharacterTextSplitter(
        separator="\n## ",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.split_text(knowledge_base_text)
    docs = [docs[0]] + ["## " + doc for doc in docs[1:]]
    print(f"Split document into {len(docs)} chunks.")

    try:
        # Using a popular, lightweight, and multilingual model
        model_name = "all-MiniLM-L6-v2"
        # The model will be downloaded automatically on the first run
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        print(f"Initialized local embeddings model: {model_name}")
    except Exception as e:
        print(f"Error initializing local embeddings model: {e}")
        return

    try:
        vector_store = Chroma.from_texts(
            texts=docs, 
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
        print(f"Vector index successfully built and saved to: {CHROMA_DB_PATH}")
    except Exception as e:
        print(f"Error creating or saving Chroma index: {e}")

def load_vector_index():
    """
    Loads the pre-built Chroma vector index from the local path.
    """
    print("Loading Chroma vector index...")
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"Error: Chroma DB not found at {CHROMA_DB_PATH}")
        print("Please run `build_vector_index()` first.")
        return None
    
    try:
        model_name = "all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH, 
            embedding_function=embeddings
        )
        print("Vector index loaded successfully.")
        return vector_store
    except Exception as e:
        print(f"Error loading Chroma index: {e}")
        return None

if __name__ == '__main__':
    print("Running vector search script directly to build the Chroma index with a local model.")
    build_vector_index()

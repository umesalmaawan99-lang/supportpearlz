import os
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from src.config import settings
from src.utils.logging_setup import setup_logging
from src.ingestion.loaders import load_documents_from_directory
from src.ingestion.chunking import get_text_splitter

logger = setup_logging()

def build_vector_index():
    logger.info("Starting knowledge base ingestion and indexing pipeline...")
    
    # 1. Load documents
    docs = load_documents_from_directory("data/knowledge_base")
    if not docs:
        logger.warning("No documents found in knowledge base. Aborting index build.")
        return
        
    # 2. Chunk documents
    splitter = get_text_splitter()
    chunks = splitter.split_documents(docs)
    logger.info(f"Split {len(docs)} documents into {len(chunks)} chunks.")
    
    # 3. Initialize Embeddings
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key
    )
    
    # 4. Create and persist FAISS vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    os.makedirs(settings.vector_store_path, exist_ok=True)
    vector_store.save_local(settings.vector_store_path)
    
    logger.info(f"Successfully built and persisted FAISS vector index with {len(chunks)} chunks at '{settings.vector_store_path}'.")
    print(f"\n[SUCCESS] FAISS vector store built successfully with {len(chunks)} chunks!")

if __name__ == "__main__":
    build_vector_index()
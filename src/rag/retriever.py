import os
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from src.config import settings

logger = logging.getLogger("SupportPearlz.RAG.Retriever")

def get_retriever():
    """Loads the persisted FAISS vector store and returns a configured retriever."""
    if not os.path.exists(settings.vector_store_path):
        raise FileNotFoundError(f"Vector store not found at {settings.vector_store_path}. Run build_index.py first.")
        
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key
    )
    
    vector_store = FAISS.load_local(
        settings.vector_store_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Configure retriever with similarity search and k=3 chunks
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    logger.info("Retriever successfully initialized from FAISS vector store.")
    return retriever

if __name__ == "__main__":
    retriever = get_retriever()
    test_query = "What are your support hours?"
    results = retriever.invoke(test_query)
    print(f"\nTest Query: '{test_query}'")
    print(f"Retrieved {len(results)} relevant chunks:")
    for i, doc in enumerate(results):
        print(f"\n--- Chunk {i+1} ---")
        print(doc.page_content)
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("CRITICAL: OPENAI_API_KEY is missing from environment variables or .env file.")
        
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.temperature = float(os.getenv("TEMPERATURE", "0.0"))
        self.vector_store_path = os.getenv("VECTOR_STORE_PATH", "data/vector_store")
        self.collection_name = os.getenv("COLLECTION_NAME", "pearlz_kb")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "120"))
        self.retrieval_k = int(os.getenv("RETRIEVAL_K", "4"))
        self.score_threshold = float(os.getenv("SCORE_THRESHOLD", "0.70"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()

if __name__ == "__main__":
    print("Configuration loaded successfully!")
    print(f"Model: {settings.llm_model}")
    print(f"Embedding Model: {settings.embedding_model}")
    print(f"Vector Store Path: {settings.vector_store_path}")
    print(f"Chunk Size: {settings.chunk_size} | Overlap: {settings.chunk_overlap}")
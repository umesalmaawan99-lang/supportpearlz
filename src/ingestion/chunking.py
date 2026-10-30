from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings

def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
    )
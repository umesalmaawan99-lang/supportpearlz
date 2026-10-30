import os
import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader
)

logger = logging.getLogger("SupportPearlz.Ingestion.Loaders")

def load_documents_from_directory(directory_path: str) -> List[Document]:
    """Loads all supported documents (.txt, .md, .pdf, .docx) from the given directory."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        logger.warning(f"Directory {directory_path} did not exist. Created empty directory.")
        return []

    documents: List[Document] = []
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            
            try:
                if ext in [".txt", ".md"]:
                    loader = TextLoader(file_path, encoding="utf-8")
                    documents.extend(loader.load())
                elif ext == ".pdf":
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                elif ext == ".docx":
                    loader = Docx2txtLoader(file_path)
                    documents.extend(loader.load())
                else:
                    logger.debug(f"Skipping unsupported file extension: {file}")
            except Exception as e:
                logger.error(f"Failed to load file {file_path}: {e}")
                
    logger.info(f"Successfully loaded {len(documents)} raw document pages/sections from {directory_path}")
    return documents

if __name__ == "__main__":
    docs = load_documents_from_directory("data/knowledge_base")
    print(f"Loaded {len(docs)} documents.")
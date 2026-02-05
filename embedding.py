from __future__ import annotations
from typing import Dict
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_models() -> Dict[str, HuggingFaceEmbeddings]:
    """Mendefinisikan Model Embedding yang akan diuji untuk eksperimen."""
    
    print("\nMemuat Model Embedding")
    models = {
        # Model BGE Small
        "Model BGE Small": HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5"), 
        
        # Model MiniLM L6
        "Model MiniLML6": HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"), 
    }
    print(f"{len(models)} Model Embedding siap diuji.")
    return models

if __name__ == "__main__":
    get_embedding_models()
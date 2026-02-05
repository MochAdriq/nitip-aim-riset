from __future__ import annotations
import os
import shutil
from typing import Dict, List
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# KONFIGURASI
VECTOR_DB_DIR = "vector_stores"

def run_indexing_experiments(chunk_sets: Dict[str, List[Document]], embedding_models: Dict) -> Dict:
    """Mengindeks semua kombinasi Chunking x Embedding ke Vector Store terpisah."""
    
    if os.path.exists(VECTOR_DB_DIR):
        shutil.rmtree(VECTOR_DB_DIR) 
    os.makedirs(VECTOR_DB_DIR)

    experiments_metadata = {}

    for chunk_key, documents in chunk_sets.items():
        for model_name, embedding_model in embedding_models.items():
            experiment_name = f"{chunk_key}_{model_name}"
            db_path = os.path.join(VECTOR_DB_DIR, experiment_name)
            
            print(f"\nMengindeks Eksperimen: {experiment_name}")
            
            try:
                Chroma.from_documents(
                    documents=documents, 
                    embedding=embedding_model, 
                    persist_directory=db_path
                )
                experiments_metadata[experiment_name] = {
                    "db_path": db_path,
                    "embedding_model": embedding_model,
                }
            except Exception as e:
                print(f"Gagal mengindeks {experiment_name}. Error: {e}")
                
    return experiments_metadata

if __name__ == "__main__":
    pass
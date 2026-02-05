from __future__ import annotations
from typing import List, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_chunking_experiments(documents: List[Document]) -> Dict[str, List[Document]]:
    """
    Melakukan eksperimen chunking RAG dengan berbagai ukuran.
    Mengembalikan dictionary berisi set chunks eksperimental.
    """
    
    # Chunk Size 256, 512, 1024
    chunk_sizes = [256, 512, 1024] 
    all_experiments: Dict[str, List[Document]] = {}

    for size in chunk_sizes:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=50, 
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents) 
        experiment_key = f'chunk_size_{size}'
        all_experiments[experiment_key] = chunks
        
        print(f"Dibuat {len(chunks)} chunks untuk konfigurasi: {experiment_key}")
        
    return all_experiments

if __name__ == "__main__":
    pass
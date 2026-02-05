from __future__ import annotations
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# Konfigurasi Data
PDF_FILE_NAME = "Dataset.pdf" 
TOPIC = "Penyakit Ringan" 
SOURCE = "Halodoc"

def load_documents(pdf_path: str, topic_name: str, source_name: str) -> List[Document]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Error: File PDF tidak ditemukan di jalur: {pdf_path}")
    
    print(f"Memuat dokumen dari: {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    for doc in documents:
        doc.metadata['topic'] = topic_name
        doc.metadata['source'] = source_name
        doc.metadata['filename'] = os.path.basename(pdf_path) 
        
    print(f"Berhasil memuat {len(documents)} halaman/dokumen.")
    return documents

if __name__ == "__main__":
    try:
        loaded_docs = load_documents(PDF_FILE_NAME, TOPIC, SOURCE)
        print("Loading dokumen selesai.")
    except Exception as e:
        print(f"Error: {e}")
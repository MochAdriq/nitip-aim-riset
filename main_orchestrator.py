from __future__ import annotations
import os
import sys
import pandas as pd
from typing import Dict, List, Any
from dotenv import load_dotenv
import os
load_dotenv()

# Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loading import load_documents, PDF_FILE_NAME, TOPIC, SOURCE
from chunking import create_chunking_experiments
from embedding import get_embedding_models
from indexing import run_indexing_experiments
from ragas_evaluation import load_test_dataset, run_ragas_evaluation, generate_final_report, TEST_DATASET_PATH


def main_workflow():

    print("Memulai Alur Kerja Eksperimen : Optimasi Kualitas Data RAG")

    try:
        # LOADING
        print("\nLoading Dokumen")
        loaded_docs = load_documents(PDF_FILE_NAME, TOPIC, SOURCE)

        # CHUNKING
        print("\nEksperimen Chunking")
        chunk_sets = create_chunking_experiments(loaded_docs)

        # EMBEDDING
        print("\nLoading Model Embedding")
        embedding_models = get_embedding_models()

        # INDEXING
        print("\nPengindeksan Eksperimental")
        metadata_eksperimen = run_indexing_experiments(chunk_sets, embedding_models)
        
        # Cek apakah ada Vector Store yang berhasil dibuat
        if not metadata_eksperimen:
            print("\nProses dihentikan: Tidak ada Vector Store yang berhasil diindeks. Periksa error sebelumnya.")
            return

        # DATASET
        print("\nMemuat Dataset Pengujian Konteks")
        test_data = load_test_dataset(TEST_DATASET_PATH)

        # RAGAS EVALUATION
        print("\nMenjalankan Evaluasi RAGAS")
        final_scores_df = run_ragas_evaluation(metadata_eksperimen, test_data)
        
        # LAPORAN AKHIR
        print("\nAnalisis Hasil dan Laporan Akhir")
        generate_final_report(final_scores_df)

    except FileNotFoundError as e:
        print(f"Gagal Total (File Tidak Ditemukan): {e}")
        print("Pastikan 'Dataset.pdf' dan 'test_dataset.json' ada di folder yang sama.")
    except Exception as e:
        print(f"Gagal Total: Terjadi Error Fatal dalam Proses Eksperimen: {e}")

if __name__ == "__main__":
    main_workflow()
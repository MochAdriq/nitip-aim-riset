from __future__ import annotations
import json
import os
import pandas as pd
import time
from typing import Dict, List
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_recall, context_precision
from ragas.run_config import RunConfig
from langchain_chroma import Chroma
# GANTI: Dari Google ke Ollama
from langchain_ollama import ChatOllama, OllamaEmbeddings 
from dotenv import load_dotenv
from tqdm import tqdm

# Load Environment Variables (Sebenarnya tidak butuh API Key lagi, tapi biarkan saja)
load_dotenv()

# Konfigurasi Nama File Dataset
TEST_DATASET_PATH = "test_dataset.json" 

def load_test_dataset(file_path: str):
    """Memuat dataset pengujian dari JSON."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} tidak ditemukan!")
    with open(file_path, 'r') as f:
        return json.load(f)

def run_ragas_evaluation(experiments_metadata: Dict, test_dataset: List[Dict]) -> pd.DataFrame:
    """
    Menjalankan RAGAS dengan OLLAMA LOKAL (Stabil & Gratis).
    """
    
    # --- KONFIGURASI OLLAMA ---
    # Model Juri: Llama 3.1 (Pintar & Cukup Ringan)
    # temperature=0: Agar konsisten dalam menilai
    llm = ChatOllama(
        model="llama3.1", 
        temperature=0
    )
    
    # Embedding Juri: Llama 3.1 juga (biar satu paket)
    embeddings = OllamaEmbeddings(
        model="llama3.1"
    )
    
    # Konfigurasi Run
    # Karena jalan di CPU i3, kita set workers=1 biar CPU tidak hang/panas berlebih.
    # Satu per satu saja, yang penting selesai.
    run_config = RunConfig(
        timeout=1200,   # Waktu tunggu sangat lama (20 menit) biar aman
        max_retries=3, 
        max_workers=1   # Single worker agar laptop Boss masih bisa dipakai kerja lain
    )
    
    all_scores = {}
    
    # Persiapan Data
    ragas_questions = [item['question'] for item in test_dataset]
    ragas_gt = []
    for item in test_dataset:
        gt = item.get('ground_truth_context', "")
        ragas_gt.append("\n".join(gt) if isinstance(gt, list) else str(gt))

    print(f"\n🚀 Memulai Evaluasi dengan OLLAMA (Lokal)...")

    # Loop Eksperimen
    experiment_bar = tqdm(experiments_metadata.items(), desc="Total Progres", unit="exp", position=0)

    for exp_name, meta in experiment_bar:
        experiment_bar.set_description(f"Menguji: {exp_name}")
        
        try:
            # 1. Retrieval (Tetap menggunakan ChromaDB lokal)
            vector_store = Chroma(
                persist_directory=meta["db_path"], 
                embedding_function=meta["embedding_model"]
            )
            
            retrieved_contexts = []
            
            # Tampilkan Jarak Sementara
            tqdm.write(f"\n--- 🔍 Retrieval: {exp_name} ---")
            
            # Gunakan k=3 agar konteks tidak terlalu panjang buat CPU
            retriever = vector_store.as_retriever(search_kwargs={"k": 3}) 
            
            for i, q in enumerate(tqdm(ragas_questions, desc="Retrieving", leave=False, position=1)):
                docs_with_scores = vector_store.similarity_search_with_score(q, k=3)
                
                docs = [doc for doc, score in docs_with_scores]
                scores = [score for doc, score in docs_with_scores]
                best_score = min(scores) if scores else 0
                
                retrieved_contexts.append([doc.page_content for doc in docs])
                
                q_short = (q[:30] + '..') if len(q) > 30 else q
                tqdm.write(f"   Q{i+1}: {q_short:<35} | Jarak: {best_score:.4f}")

            tqdm.write("   -------------------------------------------------")

            # Buat Dataset Ragas
            data_dict = {
                    "question": ragas_questions,
                    "contexts": retrieved_contexts,
                    "ground_truth": ragas_gt,
                    "answer": [""] * len(ragas_questions)
            }
            evaluation_dataset = Dataset.from_dict(data_dict)
            
            # 2. Evaluasi (OLLAMA CALL)
            tqdm.write(f"   🦙 Menilai dengan Ollama Llama 3.1 (Mungkin agak lama, santai saja)...")
            
            result = evaluate(
                dataset=evaluation_dataset,
                metrics=[context_recall, context_precision],
                llm=llm,
                embeddings=embeddings,
                run_config=run_config
            )
            
            scores = result.to_pandas().mean(numeric_only=True).fillna(0).to_dict()
            all_scores[exp_name] = scores
            
            tqdm.write(f"   ✅ Selesai! Recall={scores.get('context_recall', 0):.3f}, Precision={scores.get('context_precision', 0):.3f}\n")
            
            # Tidak perlu sleep lama-lama karena ini lokal
            time.sleep(1)

        except Exception as e:
            tqdm.write(f"   ❌ Gagal pada {exp_name}. Error: {e}")
            all_scores[exp_name] = {"context_recall": 0.0, "context_precision": 0.0}

    return pd.DataFrame(all_scores).T

def generate_final_report(scores_df: pd.DataFrame):
    """Analisis akhir."""
    print("\n=== 📊 Laporan Akhir Analisis Komparatif ===")
    
    if not scores_df.empty:
        scores_df['average_score'] = scores_df.mean(axis=1)
        print(scores_df.sort_values(by='average_score', ascending=False)) 
        
        optimal_config = scores_df['average_score'].idxmax()
        print(f"\n🏆 KONFIGURASI PEMENANG: {optimal_config}")
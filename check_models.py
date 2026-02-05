import os
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: API Key tidak ditemukan di file .env")
else:
    # 2. Konfigurasi Google AI
    genai.configure(api_key=api_key)

    print("=== DAFTAR MODEL YANG TERSEDIA ===")
    try:
        # 3. Minta daftar model dari server Google
        for m in genai.list_models():
            # Filter hanya model yang bisa generate text (bukan embedding doang)
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
                
    except Exception as e:
        print(f"Terjadi kesalahan koneksi: {e}")
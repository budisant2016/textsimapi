# app.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from docx import Document
import base64
import json
import io
import matplotlib.pyplot as plt
import seaborn as sns

# Inisialisasi FastAPI
app = FastAPI(title="Document Similarity API")

# Setup CORS untuk memungkinkan akses dari website lain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Izinkan semua origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Set NLTK data path untuk HuggingFace Space
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

# Download NLTK data explicitly
def download_nltk_data():
    try:
        nltk.download('punkt', download_dir=nltk_data_path)
        nltk.download('stopwords', download_dir=nltk_data_path)
        return True
    except Exception as e:
        print(f"Error downloading NLTK data: {str(e)}")
        return False

# Download NLTK data at startup
success = download_nltk_data()

class TextItem(BaseModel):
    content: str
    name: str

class SimilarityRequest(BaseModel):
    texts: List[TextItem]

class SimilarityResponse(BaseModel):
    similarity_matrix: List[List[float]]
    file_names: List[str]
    heatmap_base64: str

def read_docx(file_content):
    """
    Fungsi untuk membaca file DOCX dan mengekstrak teksnya
    """
    doc = Document(io.BytesIO(file_content))
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)

def simple_tokenize(text):
    """
    Fungsi tokenisasi sederhana sebagai fallback jika NLTK gagal
    """
    return text.lower().split()

def preprocess_text(text):
    """
    Fungsi untuk membersihkan dan memproses teks
    """
    # Ubah ke lowercase
    text = text.lower()
    
    # Hapus karakter khusus dan angka
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    try:
        # Coba gunakan NLTK tokenizer
        tokens = word_tokenize(text)
        # Coba gunakan NLTK stopwords
        stop_words = set(stopwords.words('indonesian'))
    except:
        # Fallback ke tokenisasi sederhana jika NLTK gagal
        tokens = simple_tokenize(text)
        # Stopwords sederhana untuk bahasa Indonesia
        stop_words = set(['yang', 'di', 'ke', 'dari', 'pada', 'dalam', 'untuk', 'dengan', 'dan', 'atau', 'ini', 'itu', 'juga', 'dari', 'saya', 'anda', 'dia', 'mereka', 'kita', 'akan', 'bisa', 'ada', 'tidak', 'saat', 'oleh', 'setelah', 'tentang', 'seperti', 'ketika', 'bagi', 'sampai', 'karena', 'jika', 'namun', 'saat', 'oleh', 'setelah'])
    
    # Hapus stopwords
    tokens = [token for token in tokens if token not in stop_words]
    
    # Gabung kembali tokens menjadi string
    return ' '.join(tokens)

def calculate_similarity_matrix(documents):
    """
    Fungsi untuk menghitung matrix kemiripan antara semua dokumen
    """
    # Preprocess semua dokumen
    processed_docs = [preprocess_text(doc) for doc in documents]
    
    # Vectorize dokumen menggunakan TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_docs)
    
    # Hitung similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    return similarity_matrix

def generate_heatmap(similarity_matrix, file_names):
    """
    Fungsi untuk menghasilkan heatmap kemiripan dan mengembalikannya sebagai base64
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(similarity_matrix, 
               annot=True, 
               fmt='.1%', 
               cmap='YlOrRd', 
               xticklabels=file_names,
               yticklabels=file_names)
    plt.title("Heatmap Kemiripan Dokumen")
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    # Convert to base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    Halaman beranda yang menampilkan dokumentasi API sederhana
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>API Matrix Kemiripan Dokumen</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 800px; margin: 0 auto; }
            h1 { color: #333; }
            .endpoint { background: #f4f4f4; padding: 10px; margin-bottom: 20px; border-radius: 5px; }
            code { background: #eee; padding: 2px 5px; }
        </style>
    </head>
    <body>
        <h1>API Matrix Kemiripan Dokumen</h1>
        <p>API ini menyediakan layanan untuk menganalisis kemiripan antar dokumen. Berikut adalah endpoint yang tersedia:</p>
        
        <div class="endpoint">
            <h2>1. Analisis Teks</h2>
            <p><strong>Endpoint:</strong> <code>POST /analyze-text</code></p>
            <p><strong>Deskripsi:</strong> Menganalisis kemiripan antara teks yang diberikan</p>
            <p><strong>Format Request:</strong></p>
            <pre><code>
{
  "texts": [
    {
      "content": "Isi dokumen pertama",
      "name": "Dokumen 1"
    },
    {
      "content": "Isi dokumen kedua",
      "name": "Dokumen 2"
    }
  ]
}
            </code></pre>
        </div>
        
        <div class="endpoint">
            <h2>2. Analisis File DOCX</h2>
            <p><strong>Endpoint:</strong> <code>POST /analyze-docx</code></p>
            <p><strong>Deskripsi:</strong> Menganalisis kemiripan antara file DOCX yang diunggah</p>
            <p><strong>Format Request:</strong> Form data dengan field <code>files</code> (multiple files)</p>
        </div>
        
        <div class="endpoint">
            <h2>3. Analisis Campuran (Teks dan File)</h2>
            <p><strong>Endpoint:</strong> <code>POST /analyze-mixed</code></p>
            <p><strong>Deskripsi:</strong> Menganalisis kemiripan antara kombinasi teks dan file DOCX</p>
            <p><strong>Format Request:</strong> Form data dengan field <code>text_data</code> (JSON string) dan <code>files</code> (opsional)</p>
        </div>
        
        <p>Untuk informasi lebih lanjut dan dokumentasi lengkap, kunjungi <a href="/docs">/docs</a>.</p>
    </body>
    </html>
    """
    return html_content

@app.post("/analyze-text", response_model=SimilarityResponse)
async def analyze_text(request: SimilarityRequest):
    """
    Endpoint untuk menganalisis kemiripan dari teks yang diberikan
    """
    if len(request.texts) < 2:
        raise HTTPException(status_code=400, detail="Minimal 2 dokumen diperlukan untuk analisis kemiripan")
    
    documents = [item.content for item in request.texts]
    file_names = [item.name for item in request.texts]
    
    # Calculate similarity matrix
    similarity_matrix = calculate_similarity_matrix(documents)
    
    # Generate heatmap
    heatmap_base64 = generate_heatmap(similarity_matrix, file_names)
    
    return {
        "similarity_matrix": similarity_matrix.tolist(),
        "file_names": file_names,
        "heatmap_base64": heatmap_base64
    }

@app.post("/analyze-docx", response_model=SimilarityResponse)
async def analyze_docx(files: List[UploadFile] = File(...)):
    """
    Endpoint untuk menganalisis kemiripan dari file DOCX
    """
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Minimal 2 dokumen diperlukan untuk analisis kemiripan")
    
    documents = []
    file_names = []
    
    for file in files:
        if not file.filename.endswith('.docx'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} bukan file DOCX")
        
        try:
            content = await file.read()
            text = read_docx(content)
            documents.append(text)
            file_names.append(file.filename)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error membaca file {file.filename}: {str(e)}")
    
    # Calculate similarity matrix
    similarity_matrix = calculate_similarity_matrix(documents)
    
    # Generate heatmap
    heatmap_base64 = generate_heatmap(similarity_matrix, file_names)
    
    return {
        "similarity_matrix": similarity_matrix.tolist(),
        "file_names": file_names,
        "heatmap_base64": heatmap_base64
    }

@app.post("/analyze-mixed")
async def analyze_mixed(
    text_data: str = Form(...),
    files: List[UploadFile] = File(None)
):
    """
    Endpoint untuk menganalisis kemiripan dari kombinasi teks dan file DOCX
    """
    try:
        # Parse text data
        text_items = json.loads(text_data)
        
        documents = []
        file_names = []
        
        # Process text data
        for item in text_items:
            documents.append(item["content"])
            file_names.append(item["name"])
        
        # Process DOCX files if any
        if files:
            for file in files:
                if not file.filename.endswith('.docx'):
                    raise HTTPException(status_code=400, detail=f"File {file.filename} bukan file DOCX")
                
                try:
                    content = await file.read()
                    text = read_docx(content)
                    documents.append(text)
                    file_names.append(file.filename)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Error membaca file {file.filename}: {str(e)}")
        
        if len(documents) < 2:
            raise HTTPException(status_code=400, detail="Minimal 2 dokumen diperlukan untuk analisis kemiripan")
            
        # Calculate similarity matrix
        similarity_matrix = calculate_similarity_matrix(documents)
        
        # Generate heatmap
        heatmap_base64 = generate_heatmap(similarity_matrix, file_names)
        
        return {
            "similarity_matrix": similarity_matrix.tolist(),
            "file_names": file_names,
            "heatmap_base64": heatmap_base64
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Data teks tidak valid (format JSON)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error dalam proses analisis: {str(e)}")

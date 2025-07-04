import requests
import numpy as np
import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer
import PyPDF2  

def download_text(url: str, filename: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    with open(filename, "w", encoding="utf-8") as f:
        f.write(response.text)
    return response.text

def extract_text_from_pdf(pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""  # Safe for pages without text
        return text

def split_text(text: str, chunk_size: int = 1024, overlap: int = 200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap  # move forward by chunk_size - overlap
    return chunks


def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def main(source: str = "url", path: str = None):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    if source == "url":
        text = download_text(
            "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt",
            "essay.txt"
        )
    elif source == "pdf" and path:
        text = extract_text_from_pdf(path)
    else:
        raise ValueError("Invalid source. Use 'url' or 'pdf' and provide a path if using PDF.")

    chunks = split_text(text, chunk_size=2048, overlap=200)
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    index = create_faiss_index(embeddings)

    faiss.write_index(index, "vector.index")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("Embedding and index creation complete.")

if __name__ == "__main__":
    # Example usage: main("pdf", "your_local_file.pdf")
    main("pdf", "Categorized list of tourist places in Odisha.pdf")  # You can change this to main("pdf", "yourfile.pdf")

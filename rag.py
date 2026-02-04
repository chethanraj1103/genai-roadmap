import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from loader import load_pdf_text, chunk_text

model = SentenceTransformer("all-MiniLM-L6-v2")

INDEX_PATH = "data/faiss.index"
DOCS_PATH = "data/docs.pkl"

index = None
docs = []


def build_index(pdf_path: str):
    global index, docs

    text = load_pdf_text(pdf_path)
    chunks = chunk_text(text, size=500, overlap=100)
    docs = chunks

    embeddings = model.encode(docs, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(docs, f)

    return len(docs)


def load_index():
    global index, docs
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(DOCS_PATH, "rb") as f:
            docs = pickle.load(f)


def retrieve(query: str, k: int = 3):
    if index is None:
        load_index()
        if index is None:
            raise RuntimeError("Index not built. Upload a PDF first.")

    q_emb = model.encode([query]).astype("float32")
    distances, indices = index.search(q_emb, k)

    results = []
    for i in indices[0]:
        if i >= 0 and i < len(docs):
            results.append(docs[i])
    return results

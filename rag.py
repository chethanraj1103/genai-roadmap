import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from loader import load_pdf_text, chunk_text

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "data/faiss.index"
META_PATH = "data/meta.pkl"

_model = None
_index = None
_chunks = None


def get_model():
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(MODEL_NAME)
        except Exception as e:
            raise RuntimeError(
                f"HF model load failed. Check HF_TOKEN or internet. Error: {e}"
            )
    return _model


def build_index_from_pdf(pdf_path: str):
    """
    Load PDF → chunk → embed → build FAISS index → persist to disk
    """
    global _index, _chunks

    text = load_pdf_text(pdf_path)
    chunks = chunk_text(text, size=500, overlap=100)

    model = get_model()
    vectors = model.encode([c["text"] for c in chunks], show_progress_bar=True)
    vectors = np.array(vectors).astype("float32")

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(chunks, f)

    _index = index
    _chunks = chunks

    return len(chunks)


def load_index():
    global _index, _chunks

    if _index is not None and _chunks is not None:
        return _index, _chunks

    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        return None, None

    _index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        _chunks = pickle.load(f)

    return _index, _chunks


def retrieve_with_sources(query: str, k: int = 3):
    """
    Query FAISS → return top-k chunks with text + metadata
    """
    index, chunks = load_index()
    if index is None:
        raise RuntimeError("Index not built yet. Upload a PDF first.")

    model = get_model()
    q_vec = model.encode([query]).astype("float32")

    distances, indices = index.search(q_vec, k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])

    return results

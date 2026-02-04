import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "data/faiss.index"
META_PATH = "data/meta.pkl"


def get_model():
    try:
        return SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print("HF model load failed:", e)
        raise RuntimeError(
            "Model download failed. Check HF_TOKEN or internet access.")


model = get_model()


def build_index(chunks):
    vectors = model.encode([c["text"] for c in chunks])
    dim = vectors.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump(chunks, f)


def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def retrieve_with_sources(query, k=3):
    index, chunks = load_index()
    q_vec = model.encode([query])

    D, I = index.search(q_vec, k)
    results = []

    for idx in I[0]:
        results.append(chunks[idx])

    return results

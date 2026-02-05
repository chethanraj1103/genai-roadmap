from pathlib import Path
from typing import List
import logging
import os
import shutil

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from groq import Groq

from rag import retrieve_with_sources, build_index_from_pdf, load_index

load_dotenv()

logger = logging.getLogger("rag_api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY not found in .env or environment variables")

client = Groq(api_key=api_key)

app = FastAPI(title="GenAI RAG API", version="1.0.0")

# Load persisted index on startup (if exists)
load_index()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class PredictResponse(BaseModel):
    answer: str
    sources: List[str]


@app.get("/", response_class=HTMLResponse)
def home():
    ui_path = Path(__file__).resolve().parent / "ui.html"
    if not ui_path.exists():
        return HTMLResponse(
            content="<h2>UI not found. Please deploy UI files.</h2>",
            status_code=500,
        )
    return FileResponse(ui_path)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        if file.content_type not in {"application/pdf"}:
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        os.makedirs("data", exist_ok=True)
        save_path = f"data/{file.filename}"

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        num_chunks = build_index_from_pdf(save_path)
        return {"status": "indexed", "chunks": num_chunks, "file": file.filename}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload/index failed")
        raise HTTPException(status_code=500, detail="Upload/index failed.") from e


def _predict(req: ChatRequest) -> PredictResponse:
    sources = retrieve_with_sources(req.message)

    # Limit context size to reduce prompt injection risk and token overflow
    max_source_chars = 2000
    trimmed_sources = [s[:max_source_chars] for s in sources]

    context_text = "\n\n".join(
        [f"[Source {i}] {s}" for i, s in enumerate(trimmed_sources)]
    )

    prompt = f"""You are an AI assistant.
Answer ONLY using the context below and cite sources like [Source 0], [Source 1].

Context:
{context_text}

Question: {req.message}
"""

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return PredictResponse(
        answer=resp.choices[0].message.content,
        sources=trimmed_sources,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: ChatRequest):
    try:
        return _predict(req)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.") from e


@app.post("/rag", response_model=PredictResponse)
def rag(req: ChatRequest):
    return predict(req)

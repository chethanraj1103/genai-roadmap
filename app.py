from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import shutil
from groq import Groq

from rag import retrieve_with_sources, build_index_from_pdf, load_index

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY not found in .env or environment variables")

client = Groq(api_key=api_key)

app = FastAPI()

# Load persisted index on startup (if exists)
load_index()


class ChatRequest(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
def home():
    ui_path = Path(__file__).parent / "ui.html"
    return ui_path.read_text(encoding="utf-8")


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        os.makedirs("data", exist_ok=True)
        save_path = f"data/{file.filename}"

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        num_chunks = build_index_from_pdf(save_path)
        return {"status": "indexed", "chunks": num_chunks, "file": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag")
def rag(req: ChatRequest):
    try:
        sources = retrieve_with_sources(req.message)

        context_text = "\n\n".join(
            [f"[Source {i}] {s}" for i, s in enumerate(sources)]
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

        return {
            "answer": resp.choices[0].message.content,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

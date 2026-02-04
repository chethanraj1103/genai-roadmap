from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import shutil
from groq import Groq

from rag import retrieve, build_index, load_index

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY not found in .env")

client = Groq(api_key=api_key)

app = FastAPI()

# Load persisted index on startup
load_index()


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": req.message}],
            temperature=0.2,
        )
        return {"reply": resp.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        os.makedirs("data", exist_ok=True)
        save_path = f"data/{file.filename}"
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        num_chunks = build_index(save_path)
        return {"status": "indexed", "chunks": num_chunks, "file": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag")
def rag(req: ChatRequest):
    try:
        context_docs = retrieve(req.message)
        prompt = f"""You are an AI assistant for BFSI fraud & risk.
Answer ONLY using the context below.

Context:
{context_docs}

Question: {req.message}
"""
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return {"context": context_docs, "reply": resp.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

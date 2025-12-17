import os, tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from dotenv import load_dotenv

from models import ChatIn, ChatOut, Citation, Verification
from rag import ingest_pdf, answer_verified

load_dotenv()
app = FastAPI(title="Enterprise RAG Platform (Verified)")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ingest/pdf")
async def ingest(tenant: str = Form(...), file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF supported in this MVP")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    n = ingest_pdf(tenant, tmp_path)
    return {"tenant": tenant, "chunks_added": n}

@app.post("/chat", response_model=ChatOut)
async def chat(inp: ChatIn):
    try:
        result = answer_verified(inp.tenant, inp.question)
    except RuntimeError as e:
        raise HTTPException(500, str(e))

    citations = []
    for d, score in result.get("docs", []):
        md = d.metadata or {}
        citations.append(Citation(
            source=str(md.get("source","unknown")),
            page=md.get("page", None),
            chunk_id=str(md.get("chunk_id","unknown")),
            score=float(score),
        ))

    v = result.get("verification", {"grounded": True, "unsupported_claims": [], "missing_evidence": []})
    verification = Verification(**v)

    return ChatOut(
        answer=result["answer"],
        citations=citations[:6],
        verification=verification,
        mode=result["mode"],
    )

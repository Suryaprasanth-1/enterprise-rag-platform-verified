# Enterprise RAG Platform (Verified Answers)

FastAPI + Chroma vector DB + OpenAI-compatible LLM endpoint.  
Includes **answer verification** (groundedness checks, citations required, similarity threshold gating, and JSON schema validation).

## Features
- PDF ingestion (per tenant) → chunking → embeddings → Chroma collection
- Chat endpoint that returns:
  - `answer`
  - `citations` (filename/page/chunk id)
  - `verification` (grounded: true/false, unsupported_claims list)
- Safety: if retrieval is weak or answer is ungrounded, returns: **"I don't know from the provided documents."**

## Tech Stack
FastAPI, LangChain, ChromaDB, OpenAI SDK (works with Groq/OpenAI-compatible endpoints)

## Run locally
### 1) API
```bash
cd api
cp .env.example .env
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2) Ingest a PDF
```bash
curl -X POST "http://localhost:8000/ingest/pdf"   -F "tenant=acme"   -F "file=@/path/to/file.pdf"
```

### 3) Ask a question
```bash
curl -X POST "http://localhost:8000/chat"   -H "Content-Type: application/json"   -d '{"tenant":"acme","question":"What is the main conclusion?"}'
```

## Environment variables
- `LLM_API_KEY` (required)
- `LLM_BASE_URL` (default: Groq OpenAI-compatible)
- `LLM_MODEL` (chat model)
- `EMBED_MODEL` (embedding model; if your provider doesn't support embeddings, set `USE_LOCAL_EMBEDDINGS=true`)
- `CHROMA_DIR` (default: ./chroma)
- `MIN_SCORE` (default: 0.25) retrieval gate for "enough evidence"
- `VERIFY_WITH_LLM` (default: true) runs an LLM groundedness check (recommended)

## Notes
- For production: replace in-process storage with managed DB + object storage, add auth, rate limiting, and structured logging.

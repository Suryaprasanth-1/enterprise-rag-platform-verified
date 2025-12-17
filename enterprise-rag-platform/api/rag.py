import os, re, json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY","")
LLM_BASE_URL = os.getenv("LLM_BASE_URL","https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL","llama-3.1-70b-versatile")

EMBED_MODEL = os.getenv("EMBED_MODEL","text-embedding-3-large")
USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS","false").lower() == "true"
CHROMA_DIR = os.getenv("CHROMA_DIR","./chroma")
MIN_SCORE = float(os.getenv("MIN_SCORE","0.25"))
VERIFY_WITH_LLM = os.getenv("VERIFY_WITH_LLM","true").lower() == "true"

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL) if LLM_API_KEY else None

def _local_embeddings():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    class E:
        def embed_documents(self, texts: List[str]):
            return model.encode(texts, normalize_embeddings=True).tolist()
        def embed_query(self, text: str):
            return model.encode([text], normalize_embeddings=True)[0].tolist()
    return E()

def _remote_embeddings():
    # LangChain's OpenAIEmbeddings uses OpenAI API style. Many OpenAI-compatible providers support it;
    # if not, set USE_LOCAL_EMBEDDINGS=true.
    from langchain_community.embeddings import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=EMBED_MODEL,
        openai_api_key=LLM_API_KEY,
        openai_api_base=LLM_BASE_URL,
    )

def embeddings():
    return _local_embeddings() if USE_LOCAL_EMBEDDINGS else _remote_embeddings()

def vectordb(tenant: str):
    return Chroma(
        collection_name=f"tenant_{tenant}",
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings(),
    )

def ingest_pdf(tenant: str, path: str) -> int:
    loader = PyPDFLoader(path)
    docs = loader.load()  # each doc has metadata incl. page
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # add stable chunk ids
    for i, c in enumerate(chunks):
        c.metadata = c.metadata or {}
        c.metadata["chunk_id"] = f"{os.path.basename(path)}::p{c.metadata.get('page',0)}::c{i}"
        c.metadata["source"] = c.metadata.get("source", os.path.basename(path))

    db = vectordb(tenant)
    db.add_documents(chunks)
    db.persist()
    return len(chunks)

SYSTEM_ANSWER = """You are a helpful assistant. Answer using ONLY the provided context.
You MUST include citations in-line like [chunk_id] for each factual claim.
If the answer is not in the context, say: "I don't know from the provided documents."
Keep the answer concise and factual.
"""

SYSTEM_VERIFY = """You are verifying whether an answer is fully supported by the given context.
Return ONLY JSON:
{
  "grounded": true/false,
  "unsupported_claims": ["..."],
  "missing_evidence": ["..."]
}
Rules:
- If any claim is not supported by the context, grounded=false.
- If citations reference chunk_ids not present, grounded=false.
- Be strict.
"""

def _extract_cited_chunk_ids(answer: str) -> List[str]:
    return re.findall(r"\[([^\]]+)\]", answer)

def _format_context(docs) -> str:
    parts=[]
    for d in docs:
        cid = d.metadata.get("chunk_id","unknown")
        src = d.metadata.get("source","unknown")
        page = d.metadata.get("page", None)
        header = f"[{cid}] source={src} page={page}"
        parts.append(header + "\n" + d.page_content)
    return "\n\n".join(parts)

def retrieve_with_scores(tenant: str, question: str, k: int = 5):
    db = vectordb(tenant)
    # Use Chroma similarity_search_with_relevance_scores when available
    try:
        results = db.similarity_search_with_relevance_scores(question, k=k)
        # returns (Document, score) where score is 0..1 relevance (higher better)
        return results
    except Exception:
        docs = db.similarity_search(question, k=k)
        return [(d, 0.5) for d in docs]  # fallback

def answer_verified(tenant: str, question: str) -> Dict[str, Any]:
    if not client:
        raise RuntimeError("LLM_API_KEY not set")

    results = retrieve_with_scores(tenant, question, k=6)
    if not results:
        return {
            "mode": "insufficient_evidence",
            "answer": "I don't know from the provided documents.",
            "docs": [],
            "verification": {"grounded": True, "unsupported_claims": [], "missing_evidence": ["No documents retrieved."]},
        }

    # evidence gate
    top_score = max([s for _, s in results])
    if top_score < MIN_SCORE:
        return {
            "mode": "insufficient_evidence",
            "answer": "I don't know from the provided documents.",
            "docs": results,
            "verification": {"grounded": True, "unsupported_claims": [], "missing_evidence": [f"Top retrieval score {top_score:.2f} < MIN_SCORE {MIN_SCORE:.2f}."]},
        }

    docs = [d for d, _ in results]
    context = _format_context(docs)

    # generate answer
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role":"system","content":SYSTEM_ANSWER},
            {"role":"user","content":f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.2,
    )
    draft = resp.choices[0].message.content.strip()

    # quick structural check: must either say don't know OR include citations
    cited = _extract_cited_chunk_ids(draft)
    if "I don't know from the provided documents" not in draft and len(cited) == 0:
        # one repair attempt: force citations
        repair = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role":"system","content":SYSTEM_ANSWER},
                {"role":"user","content":f"Your answer MUST include citations like [chunk_id] for every claim. Rewrite with citations only.\n\nContext:\n{context}\n\nQuestion: {question}\n\nDraft: {draft}"}
            ],
            temperature=0.1,
        )
        draft = repair.choices[0].message.content.strip()
        cited = _extract_cited_chunk_ids(draft)

    verification = {"grounded": True, "unsupported_claims": [], "missing_evidence": []}
    mode = "grounded"

    if VERIFY_WITH_LLM:
        # verify groundedness strictly
        v = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role":"system","content":SYSTEM_VERIFY},
                {"role":"user","content":f"Context:\n{context}\n\nAnswer:\n{draft}"}
            ],
            temperature=0.0,
            response_format={"type":"json_object"},
        )
        verification = json.loads(v.choices[0].message.content)

    # also verify citations refer to known chunk ids
    known = {d.metadata.get("chunk_id","") for d in docs}
    bad_cites = [c for c in cited if c not in known]
    if bad_cites:
        verification["grounded"] = False
        verification.setdefault("unsupported_claims", []).append(f"Answer references unknown chunk_ids: {bad_cites}")

    if not verification.get("grounded", True):
        mode = "ungrounded_rejected"
        draft = "I don't know from the provided documents."

    return {
        "mode": mode,
        "answer": draft,
        "docs": results,
        "verification": verification,
    }

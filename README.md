# Enterprise RAG Platform (Verified Answers)

An enterprise-grade Retrieval-Augmented Generation (RAG) chatbot that provides **document-grounded answers with verification**, preventing hallucinations through citation enforcement and evidence checks.

This system is designed to ensure that all responses are **supported by retrieved documents** and to safely handle cases where insufficient evidence is available.

---

## 🚀 Key Features

- **Document-based Question Answering**
  - Ingest PDFs and query them using a Retrieval-Augmented Generation (RAG) pipeline.
- **Answer Verification & Grounding**
  - Enforces mandatory citations for all factual claims.
  - Rejects ungrounded or hallucinated answers automatically.
- **Evidence Threshold Gating**
  - If retrieved document similarity is below a confidence threshold, the system responds with *“I don’t know from the provided documents.”*
- **Multi-tenant Architecture**
  - Supports separate knowledge bases per tenant.
- **Production-style API**
  - Clean FastAPI endpoints with structured JSON responses.

---

## 🧠 Verification Pipeline

Each user query goes through the following stages:

1. **Document Retrieval**
   - Vector similarity search over embedded document chunks.
2. **Evidence Check**
   - Ensures retrieved chunks meet a minimum similarity score.
3. **Grounded Answer Generation**
   - LLM generates answers strictly from retrieved context.
   - Inline citations are required for every claim.
4. **Groundedness Verification**
   - A secondary verification step checks whether claims are fully supported.
5. **Safe Fallback**
   - If verification fails, the system returns a safe “unknown” response.

---

## 🛠 Tech Stack

- **Backend:** FastAPI (Python)
- **LLM:** OpenAI-compatible APIs (Groq / OpenAI)
- **Vector Database:** ChromaDB
- **Embeddings:** OpenAI embeddings or local Sentence Transformers
- **Document Parsing:** PyPDF
- **Validation:** Pydantic schemas
- **Deployment:** Docker-ready

---

## 📁 Project Structure


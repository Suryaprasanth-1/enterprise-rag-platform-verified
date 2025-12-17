from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class ChatIn(BaseModel):
    tenant: str = Field(..., min_length=1)
    question: str = Field(..., min_length=3)

class Citation(BaseModel):
    source: str
    page: Optional[int] = None
    chunk_id: str
    score: float

class Verification(BaseModel):
    grounded: bool
    unsupported_claims: List[str] = []
    missing_evidence: List[str] = []

class ChatOut(BaseModel):
    answer: str
    citations: List[Citation] = []
    verification: Verification
    mode: Literal["grounded", "insufficient_evidence", "ungrounded_rejected"]

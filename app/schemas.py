"""The contract of our API. It does not depend on any provider (Groq, mock, ...).

YOUR JOB: fill in the fields of each class. Spec:

  OrderStatus      an enum with exactly 3 values: confirmed, pending_review, rejected
  ToolCall         id: str, name: str, arguments: dict
  LLMReply         content: str (default ""), tool_calls: list[ToolCall] (default empty),
                   model: str, tokens: int (default 0)
  Candidate        medicine_id: str, name: str, score: float
  IdentifyRequest  question: str, between 3 and 500 characters
  IdentifyResponse provider: str, model: str, latency_ms: float, answer: str,
                   sources: list[str], sufficient: bool, requires_review: bool,
                   top_score: float, candidates: list[Candidate]
  OrderRequest     request: str, between 3 and 500 characters
  OrderResult      status: OrderStatus, reason: str, order_id: str or None (default None),
                   pharmacy_id: str, medicine_id: str, quantity: int
  ToolRecord       tool: str, arguments: dict, result: dict
  OrderResponse    provider: str, model: str, latency_ms: float, reply: str,
                   tool_calls: list[ToolRecord], turns: int, aborted: bool

Check with:  python -m pytest tests/test_01_schemas.py -q
"""
from enum import Enum

from pydantic import BaseModel, Field


class OrderStatus(str, Enum):
    confirmed = "confirmed"
    pending_review = "pending_review"
    rejected = "rejected"


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict


class LLMReply(BaseModel):
    content: str = ""
    tool_calls: list[ToolCall] = []
    model: str
    tokens: int = 0


class Candidate(BaseModel):
    medicine_id: str
    name: str
    score: float


class IdentifyRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class IdentifyResponse(BaseModel):
    provider: str
    model: str
    latency_ms: float
    answer: str
    sources: list[str]
    sufficient: bool
    requires_review: bool
    top_score: float
    candidates: list[Candidate]


class OrderRequest(BaseModel):
    request: str = Field(min_length=3, max_length=500)


class OrderResult(BaseModel):
    status: OrderStatus
    reason: str
    order_id: str | None = None
    pharmacy_id: str
    medicine_id: str
    quantity: int


class ToolRecord(BaseModel):
    tool: str
    arguments: dict
    result: dict


class OrderResponse(BaseModel):
    provider: str
    model: str
    latency_ms: float
    reply: str
    tool_calls: list[ToolRecord]
    turns: int
    aborted: bool

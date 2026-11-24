import pytest
from pydantic import ValidationError

from app.schemas import (Candidate, IdentifyRequest, IdentifyResponse, LLMReply, OrderRequest,
                         OrderResponse, OrderResult, OrderStatus, ToolCall, ToolRecord)


def test_order_status_has_exactly_three_values():
    assert {s.value for s in OrderStatus} == {"confirmed", "pending_review", "rejected"}


def test_requests_have_length_limits():
    with pytest.raises(ValidationError):
        IdentifyRequest(question="ab")
    with pytest.raises(ValidationError):
        IdentifyRequest(question="x" * 501)
    with pytest.raises(ValidationError):
        OrderRequest(request="ab")
    assert IdentifyRequest(question="abc").question == "abc"
    assert OrderRequest(request="abc").request == "abc"


def test_order_result_only_accepts_known_statuses():
    ok = OrderResult(status="confirmed", reason="ok", pharmacy_id="PH001", medicine_id="M01", quantity=2)
    assert ok.order_id is None
    with pytest.raises(ValidationError):
        OrderResult(status="shipped", reason="?", pharmacy_id="PH001", medicine_id="M01", quantity=2)


def test_llm_reply_defaults():
    reply = LLMReply(model="m")
    assert reply.content == "" and reply.tool_calls == [] and reply.tokens == 0
    call = ToolCall(id="1", name="x", arguments={"a": 1})
    assert LLMReply(model="m", tool_calls=[call]).tool_calls[0].name == "x"


def test_identify_response_fields():
    cand = Candidate(medicine_id="M01", name="Cefalix 500", score=0.5)
    r = IdentifyResponse(provider="mock", model="m", latency_ms=1.0, answer="a", sources=["M01-uses"],
                         sufficient=True, requires_review=False, top_score=0.5, candidates=[cand])
    assert r.candidates[0].medicine_id == "M01"


def test_order_response_fields():
    rec = ToolRecord(tool="check_stock", arguments={"medicine_id": "M01"}, result={"stock": 1})
    r = OrderResponse(provider="mock", model="m", latency_ms=1.0, reply="ok", tool_calls=[rec],
                      turns=2, aborted=False)
    assert r.turns == 2 and r.tool_calls[0].tool == "check_stock"

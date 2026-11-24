"""Stage 3, 5 and 7 - the API. Three routes:

  GET  /health    -> {"status": "ok", "mode": ..., "provider": ...}
  POST /identify  -> Part A: describe a medicine or ask about it, get an answer with sources
  POST /order     -> Part B: a pharmacy writes a request, the agent handles it

Start it with:  uvicorn app.main:app --reload     then open http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI, HTTPException

from app.agent import run_agent
from app.config import MODE
from app.generation import answer_question
from app.providers import get_provider
from app.schemas import IdentifyRequest, IdentifyResponse, OrderRequest, OrderResponse

app = FastAPI(title="Pharmacy Leaflet Helper")

UNAVAILABLE = "Inference service unavailable"


@app.get("/health")
def health():
    return {"status": "ok", "mode": MODE, "provider": get_provider(MODE).name}


@app.post("/identify", response_model=IdentifyResponse)
def identify(body: IdentifyRequest):
    """Call answer_question(body.question). If anything goes wrong inside
    (provider down, quota...), answer HTTP 503 with the detail UNAVAILABLE, like Atelier 1."""
    try:
        return answer_question(body.question)
    except Exception:
        raise HTTPException(status_code=503, detail=UNAVAILABLE)


@app.post("/order", response_model=OrderResponse)
def order(body: OrderRequest):
    """Call run_agent(body.request). Same 503 rule as /identify."""
    try:
        return run_agent(body.request)
    except Exception:
        raise HTTPException(status_code=503, detail=UNAVAILABLE)

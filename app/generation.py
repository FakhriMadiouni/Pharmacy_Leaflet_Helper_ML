"""Stage 5 - Part A: answer a question from the leaflets, cite the sources, or say "I don't know".

Steps of the pipeline:  search -> (threshold) -> model writes -> checks.
The code, not the model, decides when to refuse.
"""
import json
import time

from app.config import MODE, SEARCH_THRESHOLD
from app.providers import get_provider
from app.schemas import Candidate, IdentifyResponse
from app.search import search_leaflets

REFUSAL = "I could not find this in the leaflets I have. Please ask a pharmacist."

SYSTEM_PROMPT = """You help people understand medicine leaflets. You are not a doctor.
Answer ONLY with the passages you are given. Each passage starts with its id in brackets.
- Never invent a dose, a delay, a contraindication or any other fact.
- If the passages do not contain the answer, set "sufficient" to false.
- Never give advice about a specific person (child, pregnancy, other medicines) unless the passages say it.
Reply with a JSON object only:
{"answer": "<one full sentence that repeats the numbers and units from the passage>", "sources": ["<passage id>", ...], "sufficient": true or false}"""


def format_passages(passages: list[dict]) -> str:
    return "\n".join(f"[{p['id']}] {p['text']}" for p in passages)


def build_candidates(passages: list[dict]) -> list[Candidate]:
    """PROVIDED. One candidate per medicine (its best passage score), best first.
    This is how the app "identifies" the medicine behind a description."""
    best = {}
    for p in passages:
        if p["medicine_id"] not in best:
            best[p["medicine_id"]] = Candidate(medicine_id=p["medicine_id"], name=p["medicine"], score=p["score"])
    return list(best.values())


def answer_question(question: str, k: int = 3, provider=None, threshold: float | None = None) -> IdentifyResponse:
    provider = provider or get_provider(MODE)
    threshold = SEARCH_THRESHOLD if threshold is None else threshold
    start = time.perf_counter()

    passages = search_leaflets(question, k)
    top_score = passages[0]["score"] if passages else 0.0
    candidates = build_candidates(passages)
    allowed_ids = {p["id"] for p in passages}

    def respond(answer, sources, sufficient, requires_review):
        return IdentifyResponse(
            provider=provider.name, model=provider.model,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            answer=answer, sources=sources, sufficient=sufficient, requires_review=requires_review,
            top_score=top_score, candidates=candidates)

    # YOUR JOB - follow these steps, using respond(...) for every return:
    #   1. top_score below threshold      -> return respond(REFUSAL, [], False, True)  WITHOUT calling the model
    #   2. call the model: provider.chat([system message, user message], json_mode=True)
    #      user message = "Passages:\n" + format_passages(passages) + "\n\nQuestion: " + question
    #   3. json.loads(reply.content); if it is not valid JSON, or it is not a dict -> refusal
    #   4. keep only the sources that are in allowed_ids (the model may invent a passage id)
    #   5. if data["sufficient"] is not True -> refusal
    #   6. otherwise return respond(data["answer"], good_sources, True, requires_review)
    #      where requires_review is True if the model invented a source or cited none.
    # Check with:  python -m pytest tests/test_05_generation.py -q
    if top_score < threshold:
        return respond(REFUSAL, [], False, True)

    reply = provider.chat(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": "Passages:\n" + format_passages(passages) + "\n\nQuestion: " + question}],
        json_mode=True)
    try:
        data = json.loads(reply.content)
    except json.JSONDecodeError:
        return respond(REFUSAL, [], False, True)
    if not isinstance(data, dict) or data.get("sufficient") is not True:
        return respond(REFUSAL, [], False, True)

    cited = data.get("sources") or []
    good_sources = [s for s in cited if s in allowed_ids]
    requires_review = len(good_sources) < len(cited) or not good_sources
    return respond(str(data.get("answer", "")), good_sources, True, requires_review)

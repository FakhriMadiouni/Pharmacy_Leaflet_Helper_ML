"""Stage 4 - measure the SEARCH alone: Recall@1, @3, @5 and MRR on the answerable questions.

Run:  python scripts/eval_search.py
No model call, no quota: only the embedding encoder is used.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.embeddings import get_encoder  # noqa: E402
from app.search import search_leaflets  # noqa: E402
from scripts.metrics import mrr, recall_at_k  # noqa: E402
from scripts.scoring import read_csv, save_results  # noqa: E402

questions = [q for q in read_csv("questions_leaflet.csv") if q["answerable"] == "1"]
expected = [q["expected_passages"].split(";") for q in questions]
found = [[p["id"] for p in search_leaflets(q["question"], 5)] for q in questions]

summary = {"recall@1": recall_at_k(expected, found, 1), "recall@3": recall_at_k(expected, found, 3),
           "recall@5": recall_at_k(expected, found, 5), "mrr": mrr(expected, found),
           "questions": len(questions)}
label = get_encoder().name
print(f"=== search: {label} ({len(questions)} answerable questions) ===")
print(f"Recall@1 : {summary['recall@1']:.3f}   Recall@3 : {summary['recall@3']:.3f}   "
      f"Recall@5 : {summary['recall@5']:.3f}   MRR : {summary['mrr']:.3f}")
print("Questions whose good passage is NOT in the top 3:")
for q, e, f in zip(questions, expected, found):
    if not set(e) & set(f[:3]):
        print(f"  {q['id']}  expected {e}  found {f[:3]}  <{q['question']}>")
items = [{"id": q["id"], "expected": e, "found": f} for q, e, f in zip(questions, expected, found)]
save_results(label, "search", summary, items)

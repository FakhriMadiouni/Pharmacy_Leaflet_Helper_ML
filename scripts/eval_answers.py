"""Stage 5 - measure the whole Part A pipeline on the 24 leaflet questions.

Run:  python scripts/eval_answers.py            (uses the threshold of your .env)
      python scripts/eval_answers.py 0.25       (try another threshold)
In real mode this calls Groq (about 1 call per question that passes the threshold).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import MODE, SEARCH_THRESHOLD  # noqa: E402
from app.generation import answer_question  # noqa: E402
from app.providers import get_provider  # noqa: E402
from scripts.scoring import facts_ok, read_csv, save_results  # noqa: E402

threshold = float(sys.argv[1]) if len(sys.argv) > 1 else SEARCH_THRESHOLD
provider = get_provider(MODE)
questions = read_csv("questions_leaflet.csv")
items = []
for q in questions:
    r = answer_question(q["question"], k=3, provider=provider, threshold=threshold)
    answerable = q["answerable"] == "1"
    items.append({"id": q["id"], "answerable": answerable, "refused": not r.sufficient,
                  "answer_ok": answerable and r.sufficient and facts_ok(r.answer, q["expected_facts"]),
                  "sources": r.sources, "top_score": r.top_score,
                  "candidates": [c.medicine_id for c in r.candidates],
                  "latency_ms": r.latency_ms, "answer": r.answer})

good = [i for i in items if i["answerable"]]
bad = [i for i in items if not i["answerable"]]
summary = {
    "provider": provider.name, "threshold": threshold,
    "good_answers": sum(i["answer_ok"] for i in good), "answerable": len(good),
    "false_refusals": sum(i["refused"] for i in good),
    "correct_refusals": sum(i["refused"] for i in bad), "to_refuse": len(bad),
}
label = f"{provider.name}__t{threshold}"
print(f"=== answers: {label} ===")
print(f"good answers     : {summary['good_answers']}/{summary['answerable']}")
print(f"false refusals   : {summary['false_refusals']}   (answerable questions we refused)")
print(f"correct refusals : {summary['correct_refusals']}/{summary['to_refuse']}")
print("Failures:")
for q, i in zip(questions, items):
    if i["answerable"] and not i["answer_ok"]:
        print(f"  {i['id']} (should answer) {'REFUSED' if i['refused'] else 'wrong facts'}  <{q['question']}>")
    if not i["answerable"] and not i["refused"]:
        print(f"  {i['id']} (should refuse)  ANSWERED  <{q['question']}>")
save_results(label, "answers", summary, items)

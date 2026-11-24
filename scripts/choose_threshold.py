"""Stage 5 - look at the top search score of every question, then choose the "I don't know" threshold.

Run:  python scripts/choose_threshold.py
Then write your choice in .env  (SEARCH_THRESHOLD=...) and explain it on your one-page sheet.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.embeddings import get_encoder  # noqa: E402
from app.search import search_leaflets  # noqa: E402
from scripts.scoring import read_csv  # noqa: E402
from scripts.threshold import threshold_table  # noqa: E402

rows = read_csv("questions_leaflet.csv")
scores = {r["id"]: search_leaflets(r["question"], 1)[0]["score"] for r in rows}
answerable = [scores[r["id"]] for r in rows if r["answerable"] == "1"]
refusal = [scores[r["id"]] for r in rows if r["answerable"] == "0"]

print(f"encoder: {get_encoder().name}")
print("Top score of each question (A = answerable, R = must be refused):")
for r in sorted(rows, key=lambda r: scores[r["id"]]):
    print(f"  {scores[r['id']]:.3f}  {'A' if r['answerable'] == '1' else 'R'}  {r['id']}  {r['question']}")

thresholds = [round(0.05 * i, 2) for i in range(0, 20)]
print(f"\n{'threshold':>9}  {'false refusals (of ' + str(len(answerable)) + ')':>24}  "
      f"{'correct refusals (of ' + str(len(refusal)) + ')':>26}")
for row in threshold_table(answerable, refusal, thresholds):
    print(f"{row['threshold']:>9.2f}  {row['false_refusals']:>24}  {row['correct_refusals']:>26}")
print("\nNote: a score threshold cannot refuse everything (some bad questions look close to a leaflet).")
print("The model's own 'sufficient' flag handles those. Measure both with eval_answers.py.")

"""Stage 5 and 7 - which STAGE failed? Builds the 2x2 table from a saved result.

Run:  python scripts/diagnose.py answers__mock__t0.2
      python scripts/diagnose.py agent__mock-rules__precise
(the name is the file in results/, without .json; `ls results` shows them)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import RESULTS  # noqa: E402
from app.search import search_leaflets  # noqa: E402
from scripts.diagnostic import diagnostic  # noqa: E402
from scripts.scoring import read_csv  # noqa: E402

data = json.loads((RESULTS / f"{sys.argv[1]}.json").read_text(encoding="utf-8"))
items = data["items"]

if data["kind"] == "answers":
    questions = {q["id"]: q for q in read_csv("questions_leaflet.csv")}
    items = [i for i in items if i["answerable"]]  # only questions that have an answer
    stage1, final = [], []
    for i in items:
        expected = set(questions[i["id"]]["expected_passages"].split(";"))
        found = {p["id"] for p in search_leaflets(questions[i["id"]]["question"], 3)}
        stage1.append(bool(expected & found))
        final.append(i["answer_ok"])
    names = ("search found the passage", "search missed it", "model wrote it right", "model wrote it wrong")
else:
    stage1 = [i["tools_ok"] for i in items]
    final = [i["facts_ok"] for i in items]
    names = ("right tools", "wrong tools", "right answer", "wrong answer")

cases = diagnostic(stage1, final)
ids = lambda positions: [items[p]["id"] for p in positions]  # noqa: E731
print(f"=== diagnostic: {sys.argv[1]} ({len(items)} items) ===")
print(f"{'':<26}{names[2]:<22} {names[3]}")
print(f"{names[0]:<26}ok: {len(cases['ok']):<19} stage2_failed: {len(cases['stage2_failed'])}")
print(f"{names[1]:<26}lucky: {len(cases['lucky']):<16} stage1_failed: {len(cases['stage1_failed'])}")
total = sum(len(v) for v in cases.values())
print(f"total of the four cells: {total}")
for case in ("stage2_failed", "lucky", "stage1_failed"):
    if cases[case]:
        print(f"  {case}: {ids(cases[case])}")

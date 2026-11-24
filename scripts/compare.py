"""Stage 8 - put all saved results side by side (PROVIDED).  Run:  python scripts/compare.py

Remember: sizes are small. One task is worth 100/24 = 4.2 points. Always say the gap in TASKS,
and do not claim one variant is better for a gap of 1 or 2 tasks.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import RESULTS  # noqa: E402

files = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS.glob("*.json"))]

def fmt(value):
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def show(kind, title, columns):
    rows = [f for f in files if f["kind"] == kind]
    if not rows:
        return
    print(f"\n{title}")
    print(f"{'label':<34}" + "".join(f"{c:>20}" for c, _ in columns))
    for f in rows:
        print(f"{f['label']:<34}" + "".join(f"{fmt(f['summary'][key]):>20}" for _, key in columns))
    if kind in ("agent", "answers"):
        key = "success" if kind == "agent" else "good_answers"
        best = max(rows, key=lambda f: f["summary"][key])
        for f in rows:
            if f is not best:
                gap = best["summary"][key] - f["summary"][key]
                print(f"  gap {best['label']} vs {f['label']}: {gap} task(s)")

show("search", "SEARCH (answerable questions)", [("Recall@1", "recall@1"), ("Recall@3", "recall@3"), ("MRR", "mrr")])
show("answers", "ANSWERS (leaflet questions)", [("good answers", "good_answers"),
     ("false refusals", "false_refusals"), ("correct refusals", "correct_refusals")])
show("agent", "AGENT (24 tasks)", [("succeeded", "success"), ("right tools", "tools_ok"),
     ("right answers", "facts_ok"), ("unrequested orders", "unrequested_orders")])

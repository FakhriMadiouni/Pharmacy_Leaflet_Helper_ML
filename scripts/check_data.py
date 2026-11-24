"""Checks that the data files agree with each other (PROVIDED).  Run:  python scripts/check_data.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import DATA  # noqa: E402
from scripts.scoring import facts_ok, read_csv  # noqa: E402

TOOLS = {"search_medicine", "check_stock", "get_pharmacy", "search_leaflet", "create_order"}


def check() -> list[str]:
    problems = []
    passages = {p["id"]: p["text"] for p in read_csv("passages.csv")}
    medicines = json.loads((DATA / "medicines.json").read_text(encoding="utf-8"))
    pharmacies = json.loads((DATA / "pharmacies.json").read_text(encoding="utf-8"))
    if len(medicines) != len(list((DATA / "notices").glob("*.md"))):
        problems.append("medicines.json and notices/ do not have the same number of medicines")
    if len(pharmacies) < 4:
        problems.append("need at least 4 pharmacies")

    questions = read_csv("questions_leaflet.csv")
    if len(questions) < 20:
        problems.append("fewer than 20 leaflet questions")
    if sum(q["answerable"] == "0" for q in questions) < 4:
        problems.append("need several questions to refuse")
    for q in questions:
        if q["answerable"] == "1":
            ids = q["expected_passages"].split(";")
            unknown = [i for i in ids if i not in passages]
            if unknown:
                problems.append(f"{q['id']}: unknown passage {unknown}")
                continue
            if not facts_ok(" ".join(passages[i] for i in ids), q["expected_facts"]):
                problems.append(f"{q['id']}: expected facts are not in the expected passages")

    tasks = read_csv("tasks_orders.csv")
    if len(tasks) < 20:
        problems.append("fewer than 20 agent tasks")
    if not any(not t["required_tools"] for t in tasks):
        problems.append("need tasks where no tool must be called")
    for t in tasks:
        bad = [x for x in t["required_tools"].split(";") if x and x not in TOOLS]
        if bad:
            problems.append(f"{t['id']}: unknown tool {bad}")
    return problems


if __name__ == "__main__":
    found = check()
    print("data OK" if not found else "\n".join(found))

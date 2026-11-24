"""How answers are scored (PROVIDED). Read it: you will have to explain these rules."""
import csv
import json
from pathlib import Path

from app.config import DATA, RESULTS

READ_ONLY = {"search_medicine", "check_stock", "get_pharmacy", "search_leaflet"}


def read_csv(name: str) -> list[dict]:
    with open(DATA / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def facts_ok(text: str, facts: str) -> bool:
    """facts = "a;b|c" means: the text must contain a AND (b OR c). Case is ignored."""
    text = text.lower()
    return all(any(alt.strip().lower() in text for alt in fact.split("|"))
               for fact in facts.split(";") if fact.strip())


def tools_ok(required: str, called: list[str], order_requested: bool) -> bool:
    """Right tools = every required tool was called, AND
         - extra READ-ONLY tools are tolerated (looking things up changes nothing),
         - create_order is only allowed if the pharmacy explicitly asked for an order,
         - when no tool is required, NO tool must be called at all."""
    need = {t for t in required.split(";") if t}
    used = set(called)
    if not need:
        return not used
    if "create_order" in used and not order_requested:
        return False
    return need <= used and not (used - need - READ_ONLY - {"create_order"})


def unrequested_orders(called: list[str], order_requested: bool) -> int:
    """Orders created although the pharmacy did not ask: a real fault, even if the answer looks right."""
    return 0 if order_requested else called.count("create_order")


def save_results(label: str, kind: str, summary: dict, items: list[dict]) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{kind}__{label}.json"
    path.write_text(json.dumps({"label": label, "kind": kind, "summary": summary, "items": items},
                               indent=2, ensure_ascii=False), encoding="utf-8")
    return path

"""Cuts every leaflet at its '## ' titles and writes data/passages.csv (PROVIDED).

This is the "preprocessing" step: one passage = one section of one leaflet,
and each passage keeps its id (e.g. M01-dosage) so we can cite it later.
Run:  python scripts/build_passages.py
"""
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import DATA  # noqa: E402

SECTION_KEYS = {
    "What it looks like": "appearance",
    "What it is used for": "uses",
    "Dosage": "dosage",
    "Warnings": "warnings",
    "Side effects": "side_effects",
}


def build() -> list[dict]:
    rows = []
    for path in sorted((DATA / "notices").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        medicine_id = path.name.split("_")[0]
        name = re.search(r"^# (.+)$", text, re.M).group(1).strip()
        active = re.search(r"\*\*Active ingredient:\*\* (\w+)", text).group(1)
        for title, body in re.findall(r"^## ([^\n]+)\n(.+?)(?=\n## |\Z)", text, re.M | re.S):
            key = SECTION_KEYS[title.strip()]
            rows.append({
                "id": f"{medicine_id}-{key}",
                "medicine_id": medicine_id,
                "medicine": name,
                "section": key,
                "text": f"{name} ({active}) - {title.strip()}: {body.strip()}",
            })
    return rows


if __name__ == "__main__":
    rows = build()
    with open(DATA / "passages.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "medicine_id", "medicine", "section", "text"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} passages written to data/passages.csv")

"""Bonus - latency of POST /identify over 20 calls: minimum, median, maximum (PROVIDED).

Run:  python scripts/latency.py
In mock mode the numbers are tiny. Run it in real mode (INFERENCE_MODE=real) for meaningful ones.
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from scripts.scoring import read_csv  # noqa: E402

client = TestClient(app)
answerable = [q["question"] for q in read_csv("questions_leaflet.csv") if q["answerable"] == "1"]
questions = (answerable * 2)[:20]  # 18 questions, repeated to make 20 calls
times = []
for question in questions:
    response = client.post("/identify", json={"question": question})
    times.append(response.json()["latency_ms"])

print(f"{len(times)} calls to /identify")
print(f"min {min(times):.1f} ms   median {statistics.median(times):.1f} ms   max {max(times):.1f} ms")

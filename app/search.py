"""Stage 4 - find the leaflet passages closest to a question.

The index is a matrix: one row per passage (60 rows), one column per vector number.
Every vector has length 1, so  index @ question_vector  gives the cosine with all passages at once.
"""
import csv
from functools import lru_cache

import numpy as np

from app.config import DATA
from app.embeddings import get_encoder


def top_k(question_vector, index, k=3):
    """Return (positions, scores) of the k passages closest to the question, best first.

    YOUR JOB. Hint: np.argsort sorts from the SMALLEST to the largest. If you forget to
    reverse it you get the 3 WORST passages and no error (the test catches it).
    Check with:  python -m pytest tests/test_03_search.py -q
    """
    scores = index @ question_vector
    positions = np.argsort(scores)[::-1][:k]
    return positions.tolist(), scores[positions].tolist()


@lru_cache(maxsize=None)
def load_index():
    """PROVIDED. Reads data/passages.csv and builds the matrix once."""
    with open(DATA / "passages.csv", encoding="utf-8") as f:
        passages = list(csv.DictReader(f))
    matrix = get_encoder().encode([p["text"] for p in passages])
    return passages, matrix


def search_leaflets(question: str, k: int = 3) -> list[dict]:
    """PROVIDED. The k closest passages, each with its score."""
    passages, matrix = load_index()
    vector = get_encoder().encode([question])[0]
    positions, scores = top_k(vector, matrix, k)
    return [{**passages[pos], "score": round(float(score), 4)} for pos, score in zip(positions, scores)]

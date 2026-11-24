import numpy as np

from app.search import search_leaflets, top_k


def test_top_k_returns_best_first():
    index = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    positions, scores = top_k(np.array([0.0, 1.0]), index, k=2)
    assert positions == [1, 2]
    assert scores == [1.0, 0.8]


def test_top_k_is_not_the_worst_passages():
    index = np.eye(5)
    positions, _ = top_k(np.array([0, 0, 1.0, 0, 0]), index, k=1)
    assert positions == [2]


def test_search_finds_the_right_leaflet():
    found = search_leaflets("How many Cefalix 500 tablets can an adult take in one day?", k=3)
    assert len(found) == 3
    assert found[0]["score"] >= found[1]["score"] >= found[2]["score"]
    assert "M01-dosage" in [p["id"] for p in found]


def test_search_gives_low_score_to_off_topic():
    assert search_leaflets("What is the best restaurant near the pharmacy?", k=1)[0]["score"] < 0.1

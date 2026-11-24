from scripts.metrics import mrr, recall_at_k

EXPECTED = [["a"], ["b"], ["c"]]
FOUND = [["a", "x", "y"], ["x", "b", "y"], ["x", "y", "z"]]


def test_recall_at_k():
    assert recall_at_k(EXPECTED, FOUND, 1) == 1 / 3
    assert recall_at_k(EXPECTED, FOUND, 2) == 2 / 3
    assert recall_at_k(EXPECTED, FOUND, 3) == 2 / 3


def test_mrr():
    assert abs(mrr(EXPECTED, FOUND) - (1 + 0.5 + 0) / 3) < 1e-9


def test_several_expected_passages_count_once():
    assert recall_at_k([["a", "b"]], [["b", "a"]], 1) == 1.0
    assert mrr([["a", "b"]], [["x", "b", "a"]]) == 0.5

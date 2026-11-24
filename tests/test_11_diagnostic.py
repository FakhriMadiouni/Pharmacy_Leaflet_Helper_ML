from scripts.diagnostic import diagnostic


def test_four_cases():
    stage1 = [True, True, False, False]
    final = [True, False, True, False]
    assert diagnostic(stage1, final) == {"ok": [0], "stage2_failed": [1], "lucky": [2], "stage1_failed": [3]}


def test_cells_add_up():
    cases = diagnostic([True, False, True, False, True], [True, True, False, False, True])
    assert sum(len(v) for v in cases.values()) == 5

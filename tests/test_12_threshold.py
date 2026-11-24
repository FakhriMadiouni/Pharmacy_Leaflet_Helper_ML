from scripts.threshold import threshold_table


def test_counts_for_each_threshold():
    rows = threshold_table(scores_answerable=[0.3, 0.5, 0.7], scores_refusal=[0.1, 0.4], thresholds=[0.2, 0.45])
    assert rows == [
        {"threshold": 0.2, "false_refusals": 0, "correct_refusals": 1},
        {"threshold": 0.45, "false_refusals": 1, "correct_refusals": 2},
    ]

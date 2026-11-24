"""Stage 5 - choosing the "I don't know" threshold from measures.

Idea: the search gives a similarity score for the best passage. Below the threshold, the app
refuses to answer WITHOUT calling the model. Too low: it answers things it should refuse.
Too high: it refuses questions it could answer.

Check with:  python -m pytest tests/test_12_threshold.py -q
"""


def threshold_table(scores_answerable, scores_refusal, thresholds):
    """For each threshold t, return a dict:
         {"threshold": t,
          "false_refusals": how many ANSWERABLE questions have a top score below t (refused by mistake),
          "correct_refusals": how many questions to REFUSE have a top score below t (refused rightly)}
    """
    rows = []
    for t in thresholds:
        rows.append({
            "threshold": t,
            "false_refusals": sum(1 for s in scores_answerable if s < t),
            "correct_refusals": sum(1 for s in scores_refusal if s < t),
        })
    return rows

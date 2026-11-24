"""Stage 4 - measure the search alone (no model, no quota).

`expected` and `found` are lists with one entry per question:
  expected[i] = ids of the good passages for question i      e.g. ["M01-dosage"]
  found[i]    = ids returned by the search, best first       e.g. ["M01-uses", "M01-dosage", "M04-dosage"]

Check with:  python -m pytest tests/test_04_metrics.py -q
"""


def recall_at_k(expected, found, k):
    """Share of questions where at least one good passage is among the first k results."""
    hits = sum(1 for e, f in zip(expected, found) if set(e) & set(f[:k]))
    return hits / len(expected)


def mrr(expected, found):
    """Mean of 1/rank of the first good passage (1 if first, 0.5 if second, 0 if absent)."""
    total = 0.0
    for e, f in zip(expected, found):
        for rank, passage_id in enumerate(f, start=1):
            if passage_id in e:
                total += 1 / rank
                break
    return total / len(expected)

"""Stage 5 and 7 - for each question, which STAGE failed?

Every pipeline of this project has two stages:
  Part A (leaflets):  1 = search finds the right passage      2 = the model writes the answer
  Part B (agent):     1 = the agent calls the right tools     2 = the final answer is right

`stage1_ok` and `final_ok` are lists of booleans, one per question. Return the POSITIONS
(indexes) of the questions in each of the four cases:

                      final answer right      final answer wrong
   stage 1 ok         "ok"                    "stage2_failed"
   stage 1 failed     "lucky"                 "stage1_failed"

"lucky" means the answer is right although stage 1 failed: the model got it by chance.

Check with:  python -m pytest tests/test_11_diagnostic.py -q
"""


def diagnostic(stage1_ok, final_ok):
    cases = {"ok": [], "stage2_failed": [], "lucky": [], "stage1_failed": []}
    for i, (s1, fin) in enumerate(zip(stage1_ok, final_ok)):
        if s1 and fin:
            cases["ok"].append(i)
        elif s1:
            cases["stage2_failed"].append(i)
        elif fin:
            cases["lucky"].append(i)
        else:
            cases["stage1_failed"].append(i)
    return cases

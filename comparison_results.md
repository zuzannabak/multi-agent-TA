# Model Comparison: Quality vs. Cost

Generated: 2026-07-15
Segment tested: `distance_metrics`
Budget ceiling: $3.00/student/week (15 segments/lecture, 1 lecture/week)

Note: token counts and assessor scores are from a single live run and will
vary slightly run to run due to normal LLM sampling. Re-run
`model_comparison.py` for fresh numbers if needed.

## Summary

| model | in tok | out tok | $/run | $/lecture | $/student/week | assessor a/b/c | budget |
|---|---|---|---|---|---|---|---|
| gemini-flash-lite-latest | 4766 | 838 | $0.00000 | $0.0000 | $0.0000 | PPF | OK |
| gpt-5.4-nano | 5310 | 1189 | $0.00255 | $0.0382 | $0.0382 | PPF | OK |
| gpt-5.4-mini | 4752 | 800 | $0.00716 | $0.1075 | $0.1075 | PPP | OK |

All three models stay well under the $3.00/student/week ceiling -- even
`gpt-5.4-mini`, the most expensive of the three, costs about $0.11/student/week,
roughly 3% of budget. Cost is not the binding constraint for any of them.

## Assessor test detail

Question: "What is the Manhattan distance between the points (1, 2) and (4, 6)?"
Expected: "7 (|1-4| + |2-6| = 3 + 4 = 7)"

### gemini-flash-lite-latest
| case | expected range | score | confidence | result |
|---|---|---|---|---|
| a: correct+confident | 0.9-1.0 | 1.0 | confident | PASS |
| b: wrong | 0.0-0.2 | 0.0 | confident | PASS |
| c: correct+hedged | 0.5-0.7 | 0.4 | hedged | **FAIL** (scored too low) |

### gpt-5.4-nano
| case | expected range | score | confidence | result |
|---|---|---|---|---|
| a: correct+confident | 0.9-1.0 | 1.0 | confident | PASS |
| b: wrong | 0.0-0.2 | 0.05 | confident | PASS |
| c: correct+hedged | 0.5-0.7 | 0.1 | guessing | **FAIL** (mislabeled as guessing, scored too low) |

### gpt-5.4-mini
| case | expected range | score | confidence | result |
|---|---|---|---|---|
| a: correct+confident | 0.9-1.0 | 0.98 | confident | PASS |
| b: wrong | 0.0-0.2 | 0.0 | confident | PASS |
| c: correct+hedged | 0.5-0.7 | 0.6 | hedged | PASS |

## Teaching text samples (first 200 chars)

**gemini-flash-lite-latest:**
> In K-Nearest Neighbors, the entire prediction relies on how you define 'near.' Because KNN identifies neighbors based on distance, your choice of metric dictates which points count as 'close,' and thi...

**gpt-5.4-nano:**
> In KNN, the prediction is built entirely from "the nearest neighbors." So the key idea is simple but powerful: **KNN depends on which points count as nearest**, and that depends on the **distance metr...

**gpt-5.4-mini:**
> KNN cannot make a prediction until you decide what counts as the "nearest" points. That means its whole prediction depends on the distance metric. In the lecture, the two practical questions are exact...

## Takeaways

- **Cost is not the constraint.** All three models land far under the
  $3.00/student/week ceiling, even at 15 segments/lecture. Budget alone
  doesn't rule out the pricier model.
- **Grading quality is the real differentiator.** Both cheaper models
  (Gemini's free tier and `gpt-5.4-nano`) under-score the hedged-but-correct
  answer (case c), treating genuine hedging almost like a wrong guess
  (0.1-0.4) instead of the intended partial-credit range (0.5-0.7).
  `gpt-5.4-nano` even mislabeled it as `"guessing"` rather than `"hedged"`.
  Only `gpt-5.4-mini` graded all three cases correctly.
- **This matters beyond prose quality:** the assessor's score writes directly
  into the knowledge-state vector that drives the planner's teach/remediate/
  advance decisions. A model that mis-grades hedged answers will misrepresent
  what a student actually understands, not just produce slightly worse
  teaching text -- that's a correctness risk, not a stylistic one.
- **Recommendation:** given the cost gap is negligible in absolute terms
  (~$0.07/student/week difference between `gpt-5.4-mini` and `gpt-5.4-nano`),
  the grading-accuracy gap favors staying on `gpt-5.4-mini` for the assessor
  role specifically, even if a cheaper model were used elsewhere.

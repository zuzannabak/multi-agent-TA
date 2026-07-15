"""
Compares LLM tiers on quality vs. cost: does a more expensive model actually
teach/grade better, against a budget ceiling of $3/week per student?

For each model, runs the SAME segment end-to-end (same persona) via the
real orchestrator/agents pipeline, and separately grades a fixed 3-case
assessor test that exposes whether cheaper models mis-grade hedged or
wrong answers -- a mis-grade there corrupts the knowledge-state vector,
which is a correctness bug, not just a quality nit.

Run:
    python model_comparison.py                     # defaults to distance_metrics
    python model_comparison.py feature_scaling
"""
import io
import sys
from contextlib import redirect_stdout
from unittest.mock import patch

import llm
import orchestrator
from agents import teacher as real_teacher, assessor as real_assessor
from knowledge_state import SEGMENTS, fresh_knowledge_state

# Windows consoles often default stdout to a legacy codepage (e.g. cp1250)
# that can't render the em-dashes/curly quotes LLMs commonly output --
# force UTF-8 so the comparison table doesn't get mangled in screenshots.
sys.stdout.reconfigure(encoding="utf-8")

SEGMENTS_PER_LECTURE = 15
LECTURES_PER_WEEK = 1
WEEKLY_BUDGET_PER_STUDENT = 3.00

MODELS = [
    {"label": "gemini-flash-lite-latest", "provider": "gemini",
     "model": "gemini-flash-lite-latest", "rate_in": 0.0, "rate_out": 0.0, "free": True},
    {"label": "gpt-5.4-nano", "provider": "openai",
     "model": "gpt-5.4-nano", "rate_in": 0.20, "rate_out": 1.25, "free": False},
    {"label": "gpt-5.4-mini", "provider": "openai",
     "model": "gpt-5.4-mini", "rate_in": 0.75, "rate_out": 4.50, "free": False},
]

# Fixed assessor test: same confirmation question for every model, so grading
# differences are attributable to the model, not the question.
ASSESSOR_TEST_QUESTION = "What is the Manhattan distance between the points (1, 2) and (4, 6)?"
ASSESSOR_TEST_EXPECTED = "7  (|1-4| + |2-6| = 3 + 4 = 7)"
ASSESSOR_TEST_CASES = [
    ("a: correct+confident", "7, since |1-4|+|2-6| = 3+4 = 7", (0.9, 1.0)),
    ("b: wrong", "5", (0.0, 0.2)),
    ("c: correct+hedged", "maybe 7? honestly not sure how this works", (0.5, 0.7)),
]

PERSONA = {
    "name": "Student B - strong math, new to ML",
    "mastery": {"linear_algebra": 0.85, "distance_metrics": 0.25},
}
PREREQ_SCORES = {
    "linear_algebra": 0.85, "calculus": 0.7, "probability_stats": 0.5,
    "big_o_analysis": 0.6, "python": 0.8,
}


class SegmentRecorder:
    """Wraps teacher()/assessor() to capture teaching text and every
    assessor "score" result during a real orchestrator.run_segment() call,
    without duplicating the orchestrator's branching logic."""

    def __init__(self):
        self.teaching_text = None
        self.scores = []

    def wrap_teacher(self, segment, task, use_rag=True):
        result = real_teacher(segment, task, use_rag=use_rag)
        if task == "teach" and self.teaching_text is None:
            self.teaching_text = result["teaching_text"]
        return result

    def wrap_assessor(self, task, **kw):
        result = real_assessor(task, **kw)
        if task == "score":
            self.scores.append(result)
        return result


def set_active_model(model_cfg):
    llm.PROVIDER = model_cfg["provider"]
    if model_cfg["provider"] == "gemini":
        llm.GEMINI_MODEL = model_cfg["model"]
    else:
        llm.OPENAI_MODEL = model_cfg["model"]


def is_quota_error(e):
    msg = str(e).lower()
    return "429" in str(e) or "resource_exhausted" in msg or "quota" in msg or "rate_limit" in msg


def run_segment_quietly(segment):
    """Run the segment end-to-end, recording teaching text + assessor
    scores, with the orchestrator's normal transcript prints suppressed
    (they'd otherwise flood the terminal three times over)."""
    recorder = SegmentRecorder()
    ks = fresh_knowledge_state(PREREQ_SCORES)
    with redirect_stdout(io.StringIO()), \
         patch.object(orchestrator, "teacher", side_effect=recorder.wrap_teacher), \
         patch.object(orchestrator, "assessor", side_effect=recorder.wrap_assessor):
        orchestrator.run_segment(segment, ks, PERSONA)
    return recorder


def run_assessor_test_quietly():
    results = []
    with redirect_stdout(io.StringIO()):
        for label, answer, expected_range in ASSESSOR_TEST_CASES:
            scored = real_assessor(
                "score",
                question=ASSESSOR_TEST_QUESTION,
                expected=ASSESSOR_TEST_EXPECTED,
                student_answer={"answer": answer},
            )
            lo, hi = expected_range
            passed = lo <= scored["score"] <= hi
            results.append({
                "label": label, "score": scored["score"],
                "confidence": scored.get("confidence"), "passed": passed,
            })
    return results


def cost_for(model_cfg, input_tokens, output_tokens):
    if model_cfg["free"]:
        return 0.0
    return (input_tokens / 1_000_000) * model_cfg["rate_in"] + \
           (output_tokens / 1_000_000) * model_cfg["rate_out"]


def evaluate_model(model_cfg, segment):
    set_active_model(model_cfg)

    llm.reset_usage()
    recorder = run_segment_quietly(segment)
    usage = llm.get_usage()

    assessor_results = run_assessor_test_quietly()

    cost_per_run = cost_for(model_cfg, usage["input_tokens"], usage["output_tokens"])
    return {
        "label": model_cfg["label"],
        "status": "ok",
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "cost_per_run": cost_per_run,
        "teaching_text": recorder.teaching_text,
        "scores": recorder.scores,
        "assessor_results": assessor_results,
    }


def print_table(rows):
    headers = ["model", "in tok", "out tok", "$/run", "$/lecture", "$/student/wk", "assessor a/b/c", "budget"]
    col_w = [26, 8, 8, 9, 10, 13, 15, 8]

    def fmt_row(cells):
        return " | ".join(str(c).ljust(w) for c, w in zip(cells, col_w))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_w))

    for r in rows:
        if r["status"] != "ok":
            print(fmt_row([r["label"], "-", "-", "-", "-", "-", "-", r["status"]]))
            continue

        per_lecture = r["cost_per_run"] * SEGMENTS_PER_LECTURE
        per_week = per_lecture * LECTURES_PER_WEEK
        abc = "".join("P" if a["passed"] else "F" for a in r["assessor_results"])
        under_budget = "OK" if per_week <= WEEKLY_BUDGET_PER_STUDENT else "OVER"

        print(fmt_row([
            r["label"],
            r["input_tokens"],
            r["output_tokens"],
            f"${r['cost_per_run']:.5f}",
            f"${per_lecture:.4f}",
            f"${per_week:.4f}",
            abc,
            under_budget,
        ]))


def print_assessor_detail(rows):
    print("\nASSESSOR TEST DETAIL (question: Manhattan distance (1,2)->(4,6), expected 7)")
    for r in rows:
        if r["status"] != "ok":
            continue
        print(f"\n  {r['label']}:")
        for a in r["assessor_results"]:
            mark = "PASS" if a["passed"] else "FAIL"
            print(f"    [{mark}] {a['label']:<24} score={a['score']:<5} confidence={a['confidence']}")


def print_teaching_samples(rows):
    print("\nTEACHING TEXT SAMPLES (first 200 chars)")
    for r in rows:
        if r["status"] != "ok":
            continue
        preview = (r["teaching_text"] or "").replace("\n", " ")[:200]
        print(f"\n  {r['label']}:\n    {preview}...")


def main():
    dim = sys.argv[1] if len(sys.argv) > 1 else "distance_metrics"
    if dim not in SEGMENTS:
        print(f"No segment defined for '{dim}'. Available: {list(SEGMENTS)}")
        return
    segment = SEGMENTS[dim]

    print(f"Comparing models on segment '{dim}' (budget ceiling: "
          f"${WEEKLY_BUDGET_PER_STUDENT:.2f}/student/week, "
          f"{SEGMENTS_PER_LECTURE} segments/lecture, {LECTURES_PER_WEEK} lecture/week)\n")

    rows = []
    for model_cfg in MODELS:
        print(f"Running {model_cfg['label']}...")
        try:
            rows.append(evaluate_model(model_cfg, segment))
        except Exception as e:
            if is_quota_error(e):
                print(f"  quota exceeded for {model_cfg['label']}, skipping.")
                rows.append({"label": model_cfg["label"], "status": "quota exceeded"})
            else:
                print(f"  error for {model_cfg['label']}: {e}")
                rows.append({"label": model_cfg["label"], "status": "error"})

    print()
    print("=" * 100)
    print("MODEL COMPARISON: QUALITY vs. COST".center(100))
    print("=" * 100)
    print_table(rows)
    print_assessor_detail(rows)
    print_teaching_samples(rows)


if __name__ == "__main__":
    main()

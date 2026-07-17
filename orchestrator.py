"""
Orchestrator: runs ONE segment end-to-end.

    planner -> if "remediate", log the weak prerequisite + a one-sentence
               bridge (never taught or re-diagnosed in-flow), THEN continue
    teach mini-unit
        -> proposer generates 0-3 concrete candidate questions
        -> if none: keep teaching (one more pass), concept stays IN_PROGRESS
        -> else judge decides ASK or SKIP
            -> SKIP: keep teaching (one more pass), concept stays IN_PROGRESS
            -> ASK: ask the chosen question -> assessor DECIDES:
                ADVANCE            -> DEMONSTRATED, move on
                RETEACH_CURRENT    -> ONE worked example, re-check once
                CHECK_PREREQUISITE -> log the gap + one-sentence bridge,
                                       NEEDS_REVIEW, continue
                REVIEW_LATER       -> NEEDS_REVIEW, continue

Design principle: the DEFAULT is to SKIP asking and keep teaching -- asking
requires the judge's justification. A concept only becomes DEMONSTRATED via
a correct answer to a concrete question, never via self-report.

The orchestrator owns turn-taking so the loop always terminates: teaching
rounds are capped (MAX_TEACH_ROUNDS), RETEACH_CURRENT only ever triggers one
retry, and a prerequisite gap gets one bridge sentence and a log entry --
never nested in-flow teaching -- so a struggling student is never stuck.

Run:
    pip install openai
    set OPENAI_API_KEY
    python orchestrator.py
"""
from knowledge_state import (
    fresh_knowledge_state, DEPENDENCY_MAP, SEGMENTS,
    NOT_SEEN, IN_PROGRESS, DEMONSTRATED, NEEDS_REVIEW,
)
from agents import planner, teacher, propose_questions, judge_questions, student, assessor

# Initial teach + at most one "keep teaching" continuation pass if nothing
# assessable has come up yet -- bounds the propose/judge loop so it always
# terminates.
MAX_TEACH_ROUNDS = 2


def set_state(ks, dimension, target_state, evidence_summary="", misconception=""):
    entry = ks[dimension]
    entry["state"] = target_state
    if evidence_summary:
        entry["evidence_summary"] = evidence_summary
    if misconception and misconception not in entry["misconceptions"]:
        entry["misconceptions"].append(misconception)
    return ks


def log(role, msg):
    print(f"\n[{role}]\n{msg}")


def run_segment(segment, knowledge_state, persona, gap_log=None):
    if gap_log is None:
        gap_log = []
    dim = segment["dimension"]
    prereqs = DEPENDENCY_MAP.get(dim, [])

    # --- Planner decides ---
    plan = planner(knowledge_state, segment, prerequisite_dims=prereqs)
    log("PLANNER", f"{plan['action']} -> {plan['target_dimension']}: {plan['rationale']}")

    # --- Planner steers: act on "remediate" instead of just logging it ---
    target = plan.get("target_dimension")
    if plan["action"] == "remediate" and target in knowledge_state and target != dim:
        knowledge_state = _remediate(target, segment, knowledge_state, gap_log)
    elif plan["action"] == "remediate":
        log("ORCHESTRATOR",
            f"Planner said remediate, but target '{target}' isn't a usable "
            f"prerequisite (same as segment or unrecognized); teaching {dim} directly.")

    # --- Teacher teaches, then propose/judge whether there's something worth asking ---
    teaching = teacher(segment, "teach")["teaching_text"]
    log("TEACHER", teaching)
    knowledge_state = set_state(knowledge_state, dim, IN_PROGRESS)
    mastery = persona["mastery"].get(dim, 0.0)

    for round_num in range(1, MAX_TEACH_ROUNDS + 1):
        proposal = propose_questions(segment, teaching)
        candidates = proposal.get("candidates", [])

        judgment = None
        if not candidates:
            log("PROPOSER", "No candidates -- nothing complete/assessable taught yet.")
        else:
            log("PROPOSER", "\n".join(
                f"- {c['question']} (reveals if wrong: {c['reveals_if_wrong']})"
                for c in candidates))
            judgment = judge_questions(candidates, segment)
            log("JUDGE", f"{judgment['action']} -- {judgment['reason']}")

        if judgment and judgment["action"] == "ASK":
            return _ask_and_decide(segment, teaching, judgment["chosen"], persona, mastery,
                                    knowledge_state, dim, prereqs, gap_log, retried=False)

        # No candidates, or judge said SKIP -- default behavior: keep teaching.
        if round_num < MAX_TEACH_ROUNDS:
            log("ORCHESTRATOR", f"Skipping the question this round; continuing to teach {dim}.")
            more = teacher(segment, "worked_example")["example_text"]
            log("TEACHER (continued)", more)
            teaching = teaching + "\n\n" + more
        else:
            log("ORCHESTRATOR",
                f"No question worth asking after {MAX_TEACH_ROUNDS} teaching rounds; "
                f"{dim} stays IN_PROGRESS.")
            return knowledge_state

    return knowledge_state  # unreachable -- loop always returns above


def _ask_and_decide(segment, teaching, chosen, persona, mastery, knowledge_state,
                     dim, prereqs, gap_log, retried):
    log("TEACHER (asking)", chosen["question"])
    sa = student(chosen["question"], teaching, persona, mastery)
    log("STUDENT", sa["answer"])

    decision = assessor("decide", dimension=dim, prerequisite_dims=prereqs,
                         question=chosen["question"], expected=chosen["expected"],
                         student_answer=sa)
    log("ASSESSOR",
        f"decision={decision['decision']} target_state={decision['target_state']}\n"
        f"  evidence: {decision['evidence']}" +
        (f"\n  misconception: {decision['misconception']}" if decision.get("misconception") else ""))

    if decision["decision"] == "ADVANCE":
        knowledge_state = set_state(knowledge_state, dim, DEMONSTRATED,
                                     decision["evidence"], decision.get("misconception", ""))
        log("ORCHESTRATOR", f"{dim} -> DEMONSTRATED, advancing.")
        return knowledge_state

    if decision["decision"] == "RETEACH_CURRENT" and not retried:
        log("ORCHESTRATOR",
            f"RETEACH_CURRENT: one worked example on {dim}, re-checking once "
            f"(single-intervention limit).")
        example = teacher(segment, "worked_example")["example_text"]
        log("TEACHER (worked example)", example)
        teaching = teaching + "\n\n" + example
        knowledge_state = set_state(knowledge_state, dim, IN_PROGRESS,
                                     decision["evidence"], decision.get("misconception", ""))
        return _ask_and_decide(segment, teaching, chosen, persona, mastery, knowledge_state,
                                dim, prereqs, gap_log, retried=True)

    if decision["decision"] == "CHECK_PREREQUISITE":
        prereq_dim = prereqs[0] if prereqs else dim
        note = decision.get("misconception") or decision["evidence"]
        gap_log.append({"topic": prereq_dim, "note": note})
        log("ORCHESTRATOR", f"Prerequisite gap flagged: '{prereq_dim}' -- {note}")
        log("TEACHER (bridge)",
            f"(Quick note: this traces back to {prereq_dim} -- keep that in mind.)")
        knowledge_state = set_state(knowledge_state, dim, NEEDS_REVIEW,
                                     decision["evidence"], decision.get("misconception", ""))
        log("ORCHESTRATOR", f"{dim} -> NEEDS_REVIEW, continuing (no nested prerequisite teaching).")
        return knowledge_state

    # REVIEW_LATER, or a RETEACH_CURRENT that already used its one intervention --
    # single-intervention limit reached; mark and continue so the student is
    # never stuck.
    knowledge_state = set_state(knowledge_state, dim, NEEDS_REVIEW,
                                 decision["evidence"], decision.get("misconception", ""))
    log("ORCHESTRATOR", f"Single-intervention limit reached; {dim} -> NEEDS_REVIEW, continuing.")
    return knowledge_state


def _remediate(target_dim, segment, knowledge_state, gap_log):
    """The planner flagged `target_dim` as a weak prerequisite for `segment`.
    Per the single-intervention design, prerequisites are never taught or
    re-diagnosed in-flow (no nested teaching) -- just logged to the running
    gap list with a one-sentence bridge, then the segment's own teaching
    proceeds."""
    note = f"Flagged weak/not-seen before teaching {segment['dimension']}."
    gap_log.append({"topic": target_dim, "note": note})
    log("ORCHESTRATOR", f"Prerequisite gap logged: '{target_dim}' -- {note}")
    log("TEACHER (bridge)",
        f"(Quick note: {segment['dimension']} builds on {target_dim} -- keep that in mind.)")
    return knowledge_state


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    # ---- Persona: strong math, weaker on the ML-specific chain ----
    # persona["mastery"] is the simulated student's TRUE grasp (0-1) of every
    # dimension, prerequisites included -- it drives how the `student` agent
    # answers. knowledge_state is the TUTOR's belief, seeded from the pretest
    # via fresh_knowledge_state() and updated only through assessor decisions.
    prereq_scores = {
        "linear_algebra": 0.35, "calculus": 0.7, "probability_stats": 0.5,
        "big_o_analysis": 0.6, "python": 0.8,
    }
    persona = {
        "name": "Student B - strong math, new to ML",
        "mastery": {
            **prereq_scores,
            "distance_metrics": 0.8,
            "feature_scaling": 0.3,   # weak -- should trigger reteach/prereq-check path
        },
    }

    ks = fresh_knowledge_state(prereq_scores, prereq_threshold=0.7)
    segment = SEGMENTS["feature_scaling"]
    gap_log = []

    print("=" * 70)
    print("RUNNING ONE SEGMENT:", segment["dimension"])
    print("=" * 70)

    ks = run_segment(segment, ks, persona, gap_log)

    print("\n" + "=" * 70)
    print("FINAL STATE for this dimension:", ks[segment["dimension"]])
    print("PREREQUISITE GAP LOG:", gap_log)
    print("=" * 70)

"""
Orchestrator: runs ONE segment end-to-end, implementing the interactive
branching diagnostic:

    planner -> if "remediate", address the named weak prerequisite first
               (teach it fully if it's its own lecture segment, otherwise
               a single diagnostic check), THEN continue below
    teach -> gate question (cheap self-report filter)
        strong_yes -> confirmation question -> assessor DECIDES:
            ADVANCE            -> DEMONSTRATED, move on
            RETEACH_CURRENT    -> worked example, re-check ONCE
            CHECK_PREREQUISITE -> prerequisite-diagnostic branch, then
                                   NEEDS_REVIEW, move on
            REVIEW_LATER       -> NEEDS_REVIEW, move on
        unsure -> worked example -> re-ask gate, then same confirmation +
                  decision step above

The orchestrator owns turn-taking so the loop always terminates: RETEACH_CURRENT
only ever triggers one retry, and CHECK_PREREQUISITE checks one level of the
dependency map, so a struggling student is never stuck on one segment.

Run:
    pip install openai
    set OPENAI_API_KEY
    python orchestrator.py
"""
from knowledge_state import (
    fresh_knowledge_state, DEPENDENCY_MAP, SEGMENTS,
    NOT_SEEN, IN_PROGRESS, DEMONSTRATED, NEEDS_REVIEW,
)
from agents import planner, teacher, student, assessor


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


def run_segment(segment, knowledge_state, persona):
    dim = segment["dimension"]
    prereqs = DEPENDENCY_MAP.get(dim, [])

    # --- Planner decides ---
    plan = planner(knowledge_state, segment, prerequisite_dims=prereqs)
    log("PLANNER", f"{plan['action']} -> {plan['target_dimension']}: {plan['rationale']}")

    # --- Planner steers: act on "remediate" instead of just logging it ---
    target = plan.get("target_dimension")
    if plan["action"] == "remediate" and target in knowledge_state and target != dim:
        knowledge_state = _remediate(target, segment, knowledge_state, persona)
    elif plan["action"] == "remediate":
        log("ORCHESTRATOR",
            f"Planner said remediate, but target '{target}' isn't a usable "
            f"prerequisite (same as segment or unrecognized); teaching {dim} directly.")

    # --- Teacher teaches ---
    teaching = teacher(segment, "teach")["teaching_text"]
    log("TEACHER", teaching)
    knowledge_state = set_state(knowledge_state, dim, IN_PROGRESS)

    # --- Gate question: cheap self-report filter (unchanged mechanism) ---
    gate_q = segment["gate_question"]
    log("TEACHER (gate)", gate_q)
    mastery = persona["mastery"].get(dim, 0.0)
    s = student(gate_q, teaching, persona, mastery)
    log("STUDENT", f"({s['self_reported_understanding']}) {s['answer']}")

    gate = assessor("evaluate_gate", gate_question=gate_q, student_answer=s)
    log("ASSESSOR", f"gate branch = {gate['branch']}")

    if gate["branch"] == "unsure":
        example = teacher(segment, "worked_example")["example_text"]
        log("TEACHER (worked example)", example)
        teaching = teaching + "\n\n" + example

        s = student(gate_q, teaching, persona, mastery)
        log("STUDENT", f"({s['self_reported_understanding']}) {s['answer']}")
        gate = assessor("evaluate_gate", gate_question=gate_q, student_answer=s)
        log("ASSESSOR", f"gate branch after example = {gate['branch']}")

    # --- Real check: confirmation question, graded by ONE assessor decision ---
    return _check_and_advance(segment, teaching, persona, mastery, knowledge_state,
                               dim, prereqs, retried=False)


def _check_and_advance(segment, teaching, persona, mastery, knowledge_state,
                        dim, prereqs, retried):
    conf = segment["confirmation"]
    log("ASSESSOR (confirmation)", conf["question"])
    sa = student(conf["question"], teaching, persona, mastery)
    log("STUDENT", sa["answer"])

    decision = assessor("decide", dimension=dim, prerequisite_dims=prereqs,
                         question=conf["question"], expected=conf["expected"],
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
        log("ORCHESTRATOR", f"RETEACH_CURRENT: worked example on {dim}, re-checking once.")
        example = teacher(segment, "worked_example")["example_text"]
        log("TEACHER (worked example)", example)
        teaching = teaching + "\n\n" + example
        knowledge_state = set_state(knowledge_state, dim, IN_PROGRESS,
                                     decision["evidence"], decision.get("misconception", ""))
        return _check_and_advance(segment, teaching, persona, mastery, knowledge_state,
                                   dim, prereqs, retried=True)

    if decision["decision"] == "CHECK_PREREQUISITE":
        knowledge_state = _check_prerequisites(segment, knowledge_state, persona)
        knowledge_state = set_state(knowledge_state, dim, NEEDS_REVIEW,
                                     decision["evidence"], decision.get("misconception", ""))
        log("ORCHESTRATOR", f"Prerequisite(s) checked; {dim} -> NEEDS_REVIEW, advancing.")
        return knowledge_state

    # REVIEW_LATER, or a RETEACH_CURRENT that already used its one retry --
    # depth limit reached, mark and move on so the student is never stuck.
    knowledge_state = set_state(knowledge_state, dim, NEEDS_REVIEW,
                                 decision["evidence"], decision.get("misconception", ""))
    log("ORCHESTRATOR", f"Depth limit reached; {dim} -> NEEDS_REVIEW, advancing.")
    return knowledge_state


def _check_prerequisites(segment, knowledge_state, persona):
    """CHECK_PREREQUISITE branch: run this segment's authored prerequisite
    diagnostics (one level deep, per the depth limit) via the dependency map
    and record what each one shows."""
    for diag in segment.get("prerequisite_diagnostics", []):
        pdim = diag["dimension"]
        pmastery = persona["mastery"].get(pdim, 0.0)
        pa = student(diag["question"], "", persona, pmastery)
        log(f"STUDENT (prereq: {pdim})", pa["answer"])

        pdecision = assessor("decide", dimension=pdim, prerequisite_dims=[],
                              question=diag["question"], expected=diag["expected"],
                              student_answer=pa)
        log("ASSESSOR",
            f"prereq {pdim}: decision={pdecision['decision']} "
            f"target_state={pdecision['target_state']} ({pdecision['evidence']})")
        knowledge_state = set_state(knowledge_state, pdim, pdecision["target_state"],
                                     pdecision["evidence"], pdecision.get("misconception", ""))
        if pdecision["target_state"] == NEEDS_REVIEW:
            log("ASSESSOR", f"GAP FOUND upstream in '{pdim}'. Remediation pointer: "
                            f"review {pdim} before revisiting {segment['dimension']}.")
    return knowledge_state


def _remediate(target_dim, segment, knowledge_state, persona):
    """The planner flagged `target_dim` as a weak prerequisite for `segment`.
    Address it before teaching the segment itself, instead of just logging
    the planner's advice and teaching through it anyway."""
    log("ORCHESTRATOR",
        f"Planner steering: remediating '{target_dim}' before teaching {segment['dimension']}.")

    if target_dim in SEGMENTS:
        # It's itself a lecture topic we have full teaching material for --
        # run the normal teach/gate/confirm loop on it first.
        return run_segment(SEGMENTS[target_dim], knowledge_state, persona)

    # No lecture segment for this dimension (e.g. a pretest-only prerequisite
    # like linear_algebra) -- fall back to this segment's own diagnostic
    # question for it, if one exists.
    diag = next((d for d in segment.get("prerequisite_diagnostics", [])
                 if d["dimension"] == target_dim), None)
    if diag is None:
        log("ORCHESTRATOR",
            f"No lecture segment or diagnostic available for '{target_dim}'; "
            f"proceeding to teach {segment['dimension']} without remediation.")
        return knowledge_state

    mastery = persona["mastery"].get(target_dim, 0.0)
    pa = student(diag["question"], "", persona, mastery)
    log(f"STUDENT (remediation: {target_dim})", pa["answer"])
    decision = assessor("decide", dimension=target_dim, prerequisite_dims=[],
                         question=diag["question"], expected=diag["expected"],
                         student_answer=pa)
    log("ASSESSOR",
        f"remediation {target_dim}: decision={decision['decision']} "
        f"target_state={decision['target_state']} ({decision['evidence']})")
    return set_state(knowledge_state, target_dim, decision["target_state"],
                      decision["evidence"], decision.get("misconception", ""))


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

    print("=" * 70)
    print("RUNNING ONE SEGMENT:", segment["dimension"])
    print("=" * 70)

    ks = run_segment(segment, ks, persona)

    print("\n" + "=" * 70)
    print("FINAL STATE for this dimension:", ks[segment["dimension"]])
    print("=" * 70)

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
    primary_conceptual_dimension,
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


def apply_dimension_updates(ks, dimension_updates, misconception=""):
    """Apply every entry in the assessor's dimension_updates list to the
    knowledge state independently -- one answer can move several dimensions
    at once, each to its own new_state, based only on the evidence the
    assessor found for that specific dimension."""
    for update in dimension_updates:
        ks = set_state(ks, update["dimension"], update["new_state"],
                        update["evidence"], misconception)
    return ks


def log_dimension_updates(decision, all_dimensions):
    updated = {u["dimension"] for u in decision["dimension_updates"]}
    print(f"\n[ASSESSOR] decision={decision['decision']}")
    for update in decision["dimension_updates"]:
        print(f"  {update['dimension']} -> {update['new_state']} ({update['evidence']})")
    for dim in all_dimensions:
        if dim not in updated:
            print(f"  ({dim}: no evidence in this answer)")
    if decision.get("misconception"):
        print(f"  misconception: {decision['misconception']}")


def log(role, msg):
    print(f"\n[{role}]\n{msg}")


def run_segment(segment, knowledge_state, persona):
    # A segment names conceptual/technical/foundational dimensions together.
    # `dim` (the Group C / conceptual one) is still used for planner
    # prerequisites, gate mastery, and logging labels, but the assessor now
    # grades against the segment's FULL dimensions list and updates each one
    # independently based on the evidence actually present in the answer.
    dim = primary_conceptual_dimension(segment["dimensions"])
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

    decision = assessor("decide", dimensions=segment["dimensions"], prerequisite_dims=prereqs,
                         question=conf["question"], expected=conf["expected"],
                         student_answer=sa)
    log_dimension_updates(decision, segment["dimensions"])
    knowledge_state = apply_dimension_updates(knowledge_state, decision["dimension_updates"],
                                               decision.get("misconception", ""))

    if decision["decision"] == "ADVANCE":
        log("ORCHESTRATOR", "Advancing.")
        return knowledge_state

    if decision["decision"] == "RETEACH_CURRENT" and not retried:
        log("ORCHESTRATOR", f"RETEACH_CURRENT: worked example on {dim}, re-checking once.")
        example = teacher(segment, "worked_example")["example_text"]
        log("TEACHER (worked example)", example)
        teaching = teaching + "\n\n" + example
        return _check_and_advance(segment, teaching, persona, mastery, knowledge_state,
                                   dim, prereqs, retried=True)

    if decision["decision"] == "CHECK_PREREQUISITE":
        knowledge_state = _check_prerequisites(segment, knowledge_state, persona)
        log("ORCHESTRATOR", "Prerequisite(s) checked; advancing.")
        return knowledge_state

    # REVIEW_LATER, or a RETEACH_CURRENT that already used its one retry --
    # depth limit reached, move on (per dimension_updates already applied
    # above) so the student is never stuck.
    log("ORCHESTRATOR", "Depth limit reached; advancing.")
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

        pdecision = assessor("decide", dimensions=[pdim], prerequisite_dims=[],
                              question=diag["question"], expected=diag["expected"],
                              student_answer=pa)
        log_dimension_updates(pdecision, [pdim])
        knowledge_state = apply_dimension_updates(knowledge_state, pdecision["dimension_updates"],
                                                    pdecision.get("misconception", ""))
        updated_states = {u["dimension"]: u["new_state"] for u in pdecision["dimension_updates"]}
        if updated_states.get(pdim) == NEEDS_REVIEW:
            log("ASSESSOR", f"GAP FOUND upstream in '{pdim}'. Remediation pointer: "
                            f"review {pdim} before revisiting {primary_conceptual_dimension(segment['dimensions'])}.")
    return knowledge_state


def _remediate(target_dim, segment, knowledge_state, persona):
    """The planner flagged `target_dim` as a weak prerequisite for `segment`.
    Address it before teaching the segment itself, instead of just logging
    the planner's advice and teaching through it anyway."""
    log("ORCHESTRATOR",
        f"Planner steering: remediating '{target_dim}' before teaching "
        f"{primary_conceptual_dimension(segment['dimensions'])}.")

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
            f"proceeding to teach {primary_conceptual_dimension(segment['dimensions'])} "
            f"without remediation.")
        return knowledge_state

    mastery = persona["mastery"].get(target_dim, 0.0)
    pa = student(diag["question"], "", persona, mastery)
    log(f"STUDENT (remediation: {target_dim})", pa["answer"])
    decision = assessor("decide", dimensions=[target_dim], prerequisite_dims=[],
                         question=diag["question"], expected=diag["expected"],
                         student_answer=pa)
    log_dimension_updates(decision, [target_dim])
    return apply_dimension_updates(knowledge_state, decision["dimension_updates"],
                                    decision.get("misconception", ""))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    # ---- Persona: strong math, weaker on the ML-specific chain ----
    # persona["mastery"] is the simulated student's TRUE grasp (0-1) of every
    # dimension, prerequisites included -- it drives how the `student` agent
    # answers. knowledge_state is the TUTOR's belief, seeded from the pretest
    # via fresh_knowledge_state() and updated only through assessor decisions.
    prereq_scores = {
        "linear_algebra": 0.35, "formula_application": 0.6, "probability_stats": 0.5,
        "computational_thinking": 0.6, "python_reading": 0.8,
    }
    persona = {
        "name": "Student B - strong math, new to ML",
        "mastery": {
            **prereq_scores,
            "distance_computation": 0.8,
            "scaling_computation": 0.3,
            "knn_intuition": 0.3,   # weak -- should trigger reteach/prereq-check path
        },
    }

    ks = fresh_knowledge_state(prereq_scores, prereq_threshold=0.7)
    segment = SEGMENTS["feature_scaling"]
    dim = primary_conceptual_dimension(segment["dimensions"])

    print("=" * 70)
    print("RUNNING ONE SEGMENT:", dim)
    print("=" * 70)

    ks = run_segment(segment, ks, persona)

    print("\n" + "=" * 70)
    print("FINAL STATE for this segment's dimensions:")
    for d in segment["dimensions"]:
        print(f"  {d}: {ks[d]}")
    print("=" * 70)

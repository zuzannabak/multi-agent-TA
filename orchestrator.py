"""
Orchestrator: runs ONE segment end-to-end, implementing the interactive
branching diagnostic:

    teach -> gate question
        strong_yes -> confirmation question -> score -> update
        unsure     -> worked example -> re-ask gate
            now yes -> confirmation -> score -> update
            still unsure -> prerequisite check (ONE level, per depth limit)
                            -> mark gap(s), give pointer, MOVE ON

The orchestrator owns turn-taking so the loop always terminates.

Run:
    pip install anthropic
    set ANTHROPIC_API_KEY
    python orchestrator.py
"""
from knowledge_state import fresh_knowledge_state, DEPENDENCY_MAP, SEGMENTS
from agents import planner, teacher, student, assessor


def update_state(ks, dimension, score):
    ks[dimension] = {"score": round(score, 2), "taught": True}
    return ks


def log(role, msg):
    print(f"\n[{role}]\n{msg}")


def run_segment(segment, knowledge_state, persona):
    dim = segment["dimension"]

    # --- Planner decides ---
    plan = planner(knowledge_state, segment)
    log("PLANNER", f"{plan['action']} -> {plan['target_dimension']}: {plan['rationale']}")

    # --- Teacher teaches ---
    teaching = teacher(segment, "teach")["teaching_text"]
    log("TEACHER", teaching)

    # --- Gate question ---
    gate_q = segment["gate_question"]
    log("TEACHER (gate)", gate_q)
    mastery = persona["mastery"].get(dim, 0.0)
    s = student(gate_q, teaching, persona, mastery)
    log("STUDENT", f"({s['self_reported_understanding']}) {s['answer']}")

    gate = assessor("evaluate_gate", gate_question=gate_q, student_answer=s)
    log("ASSESSOR", f"branch = {gate['branch']}")

    # --- STRONG YES: confirm and advance ---
    if gate["branch"] == "strong_yes":
        return _confirm_and_update(segment, teaching, persona, mastery, knowledge_state, dim)

    # --- UNSURE: worked example, then re-ask ---
    example = teacher(segment, "worked_example")["example_text"]
    log("TEACHER (worked example)", example)

    s2 = student(gate_q, teaching + "\n\n" + example, persona, mastery)
    log("STUDENT", f"({s2['self_reported_understanding']}) {s2['answer']}")

    gate2 = assessor("evaluate_gate", gate_question=gate_q, student_answer=s2)
    log("ASSESSOR", f"branch after example = {gate2['branch']}")

    if gate2["branch"] == "strong_yes":
        return _confirm_and_update(segment, teaching, persona, mastery, knowledge_state, dim)

    # --- STILL UNSURE: prerequisite check (one level deep) ---
    log("ASSESSOR", "Still unsure after example. Checking prerequisites...")
    prereq_gap = False
    for diag in segment.get("prerequisite_diagnostics", []):
        pdim = diag["dimension"]
        pmastery = persona["mastery"].get(pdim, knowledge_state[pdim]["score"])
        pa = student(diag["question"], "", persona, pmastery)
        log(f"STUDENT (prereq: {pdim})", f"{pa['answer']}")
        scored = assessor("score", question=diag["question"],
                          expected=diag["expected"], student_answer=pa)
        log("ASSESSOR", f"prereq {pdim}: score={scored['score']} ({scored['reasoning']})")
        if scored["score"] < 0.5:
            prereq_gap = True
            log("ASSESSOR", f"GAP FOUND upstream in '{pdim}'. Remediation pointer: "
                            f"review {pdim} before revisiting {dim}.")

    # Depth limit: we checked one level. Mark the gap and MOVE ON so a weak
    # student is never stuck. The low score + taught=True records the gap.
    knowledge_state = update_state(knowledge_state, dim, 0.2)
    reason = "prerequisite gap upstream" if prereq_gap else "concept unclear, no upstream cause"
    log("ORCHESTRATOR",
        f"Depth limit reached ({reason}). Marked {dim}=0.2, advancing.")
    return knowledge_state


def _confirm_and_update(segment, teaching, persona, mastery, knowledge_state, dim):
    conf = segment["confirmation"]
    log("ASSESSOR (confirmation)", conf["question"])
    sa = student(conf["question"], teaching, persona, mastery)
    log("STUDENT", sa["answer"])
    scored = assessor("score", question=conf["question"],
                      expected=conf["expected"], student_answer=sa)
    log("ASSESSOR", f"score={scored['score']} ({scored['reasoning']})")
    knowledge_state = update_state(knowledge_state, dim, scored["score"])
    log("ORCHESTRATOR", f"Updated {dim} = {scored['score']}, advancing.")
    return knowledge_state


if __name__ == "__main__":
    # ---- Persona: strong linear algebra, but shaky on distance metrics ----
    # Flip "linear_algebra" to ~0.2 to watch the prerequisite-gap branch fire.
    persona = {
        "name": "Student B - strong math, new to ML",
        "mastery": {
            "linear_algebra": 0.85,
            "distance_metrics": 0.25,   # weak on the concept itself
        },
    }
    prereq_scores = {
        "linear_algebra": 0.85, "calculus": 0.7, "probability_stats": 0.5,
        "big_o_analysis": 0.6, "python": 0.8,
    }

    ks = fresh_knowledge_state(prereq_scores)
    segment = SEGMENTS["distance_metrics"]

    print("=" * 70)
    print("RUNNING ONE SEGMENT:", segment["dimension"])
    print("=" * 70)

    ks = run_segment(segment, ks, persona)

    print("\n" + "=" * 70)
    print("FINAL STATE for this dimension:", ks[segment["dimension"]])
    print("=" * 70)

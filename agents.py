"""
The four agents. Each is a single function that builds a prompt, calls the
model, and returns structured output. The orchestrator wires them together.

Contract summary:
  planner  : reads knowledge_state           -> decides teach / remediate / advance
  teacher  : given a segment + task          -> generates teaching text or worked example
  student  : given content + question + mastery -> answers + self-reports understanding
  assessor : scores answers, decides branch  -> (orchestrator applies to knowledge_state)
"""
import json
from llm import call_json


# ---------------------------------------------------------------------------
# PLANNER
# ---------------------------------------------------------------------------
def planner(knowledge_state, segment):
    """Decide what to do with the current segment given the student's state."""
    system = (
        "You are the Planner agent in a one-on-one AI tutoring system. "
        "You read the student's knowledge state and decide the next action for a "
        "lecture segment. Return ONLY JSON with keys: "
        '"action" (one of "teach", "remediate", "advance"), '
        '"target_dimension" (string), "rationale" (one sentence). '
        "Use 'teach' if the segment's dimension has taught=false. "
        "Use 'remediate' if a prerequisite in the dependency chain is weak (score < 0.4). "
        "Use 'advance' only if the dimension is already taught with score >= 0.7."
    )
    user = (
        f"Segment dimension: {segment['dimension']}\n"
        f"Concept: {segment['concept']}\n\n"
        f"Current knowledge_state:\n{json.dumps(knowledge_state, indent=2)}"
    )
    return call_json(system, user, max_tokens=300)


# ---------------------------------------------------------------------------
# TEACHER
# ---------------------------------------------------------------------------
def teacher(segment, task):
    """task = 'teach'  -> returns {'teaching_text': ...}
       task = 'worked_example' -> returns {'example_text': ...}
    """
    if task == "teach":
        system = (
            "You are the Teacher agent. Teach ONE lecture segment clearly and "
            "concisely, as a professor would in a one-on-one setting. Cover the "
            "given key points in plain language with at most one short example. "
            "Do not quiz the student. Return ONLY JSON with key \"teaching_text\"."
        )
        user = (
            f"Concept: {segment['concept']}\n"
            f"Key points to cover:\n- " + "\n- ".join(segment["key_points"])
        )
        return call_json(system, user, max_tokens=600)

    if task == "worked_example":
        system = (
            "You are the Teacher agent. The student was unsure after the first "
            "explanation. Give ONE short, concrete worked example that makes the "
            "concept click. Do not re-teach everything; just illustrate. "
            "Return ONLY JSON with key \"example_text\"."
        )
        user = (
            f"Concept: {segment['concept']}\n"
            f"Key points:\n- " + "\n- ".join(segment["key_points"])
        )
        return call_json(system, user, max_tokens=400)

    raise ValueError(f"Unknown teacher task: {task}")


# ---------------------------------------------------------------------------
# STUDENT  (simulated learner, driven by its persona's mastery)
# ---------------------------------------------------------------------------
def student(question, context, persona, topic_mastery):
    """Answer AS a student whose grasp of this topic is `topic_mastery` (0-1).
    Low mastery should genuinely surface as confusion or wrong answers.
    Returns {'answer': ..., 'self_reported_understanding': 'yes'|'not_sure'|'no'}.
    """
    system = (
        "You are simulating a real student in a machine learning course. "
        f"Persona: {persona['name']}. "
        f"Your current grasp of THIS topic is {topic_mastery:.2f} on a 0-1 scale "
        "(0 = no idea, 1 = solid). Answer authentically at that level: if your "
        "grasp is low, be genuinely confused or make the kind of mistake a "
        "struggling student makes -- do NOT answer correctly just because you can. "
        "If your grasp is high, answer correctly and confidently. "
        "Return ONLY JSON with keys: \"answer\" (your response to the question) and "
        "\"self_reported_understanding\" (one of \"yes\", \"not_sure\", \"no\")."
    )
    user = (
        f"What the teacher just said:\n{context}\n\n"
        f"Question you are being asked:\n{question}"
    )
    return call_json(system, user, max_tokens=400)


# ---------------------------------------------------------------------------
# ASSESSOR
# ---------------------------------------------------------------------------
def assessor(task, **kw):
    """task = 'evaluate_gate' -> {'branch': 'strong_yes'|'unsure'}
       task = 'score'         -> {'correct': bool, 'score': 0-1, 'reasoning': str}
    """
    if task == "evaluate_gate":
        system = (
            "You are the Assessor agent. A student answered a 'do you understand?' "
            "gate question. Decide the branch. Return ONLY JSON with key \"branch\": "
            "\"strong_yes\" only if they clearly and confidently understand; "
            "\"unsure\" otherwise (any hesitation, 'not sure', or shaky answer)."
        )
        user = (
            f"Gate question: {kw['gate_question']}\n"
            f"Student self-report: {kw['student_answer'].get('self_reported_understanding')}\n"
            f"Student answer: {kw['student_answer'].get('answer')}"
        )
        return call_json(system, user, max_tokens=200)

    if task == "score":
        system = (
            "You are the Assessor agent. Grade the student's answer against the "
            "expected answer. Be strict but fair: partial credit is allowed. "
            "Return ONLY JSON with keys: \"correct\" (bool), \"score\" (0-1 float), "
            "\"reasoning\" (one sentence)."
        )
        user = (
            f"Question: {kw['question']}\n"
            f"Expected answer: {kw['expected']}\n"
            f"Student answer: {kw['student_answer'].get('answer')}"
        )
        return call_json(system, user, max_tokens=300)

    raise ValueError(f"Unknown assessor task: {task}")

"""
The four agents. Each is a single function that builds a prompt, calls the
model, and returns structured output. The orchestrator wires them together.

Contract summary:
  planner  : reads knowledge_state           -> decides teach / remediate / advance
  teacher  : given a segment + task          -> generates teaching text or worked example
  student  : given content + question + mastery -> answers + self-reports understanding
  assessor : evaluates gate self-report, and makes the ADVANCE / RETEACH /
             CHECK_PREREQUISITE / REVIEW_LATER call directly on a checked
             answer -> (orchestrator applies target_state to knowledge_state)
"""
import json
from llm import call_json
from retrieval import retrieve_for_segment, format_context


# ---------------------------------------------------------------------------
# PLANNER
# ---------------------------------------------------------------------------
def planner(knowledge_state, segment, prerequisite_dims=None):
    """Decide what to do with the current segment given the student's state."""
    system = (
        "You are the Planner agent in a one-on-one AI tutoring system. "
        "You read the student's categorical knowledge state and decide the "
        "next action for a lecture segment. Each topic's state is one of: "
        "NOT_SEEN, IN_PROGRESS, DEMONSTRATED, NEEDS_REVIEW. Return ONLY JSON "
        "with keys: "
        '"action" (one of "teach", "remediate", "advance"), '
        '"target_dimension" (string), "rationale" (one sentence). '
        "Use 'teach' if the segment's own dimension is NOT_SEEN or IN_PROGRESS. "
        "Use 'remediate' if one of the segment's listed prerequisites is "
        "NEEDS_REVIEW or NOT_SEEN -- target_dimension should name that "
        "prerequisite. "
        "Use 'advance' only if the segment's own dimension is already "
        "DEMONSTRATED."
    )
    user = (
        f"Segment dimension: {segment['dimension']}\n"
        f"Concept: {segment['concept']}\n"
        f"Prerequisites for this segment: {prerequisite_dims or 'none'}\n\n"
        f"Current knowledge_state:\n{json.dumps(knowledge_state, indent=2)}"
    )
    return call_json(system, user, max_tokens=300)


# ---------------------------------------------------------------------------
# TEACHER
# ---------------------------------------------------------------------------
def teacher(segment, task, use_rag=True):
    """task = 'teach'  -> returns {'teaching_text': ...}
       task = 'worked_example' -> returns {'example_text': ...}

    use_rag: if True (default), ground the response in lecture chunks
    retrieved for this segment's dimension. If False, fall back to the
    LLM's own generic knowledge -- kept as a switch for before/after
    comparison of retrieval-grounded vs. ungrounded teaching.
    """
    context_block = ""
    if use_rag:
        chunks = retrieve_for_segment(segment)
        context_block = format_context(chunks)

    rag_instruction = (
        " Teach this segment using the instructor's actual lecture material "
        "below. Use the instructor's own examples, analogies, and phrasing "
        "where they exist -- do not substitute generic textbook explanations."
    )

    if task == "teach":
        system = (
            "You are the Teacher agent. Teach ONE lecture segment clearly and "
            "concisely, as a professor would in a one-on-one setting. Cover the "
            "given key points in plain language with at most one short example. "
            "Do not quiz the student. Return ONLY JSON with key \"teaching_text\"."
        )
        if context_block:
            system += rag_instruction
        user = (
            f"Concept: {segment['concept']}\n"
            f"Key points to cover:\n- " + "\n- ".join(segment["key_points"])
        )
        if context_block:
            user += f"\n\nInstructor's lecture material:\n{context_block}"
        return call_json(system, user, max_tokens=600)

    if task == "worked_example":
        system = (
            "You are the Teacher agent. The student was unsure after the first "
            "explanation. Give ONE short, concrete worked example that makes the "
            "concept click. Do not re-teach everything; just illustrate. "
            "Return ONLY JSON with key \"example_text\"."
        )
        if context_block:
            system += rag_instruction
        user = (
            f"Concept: {segment['concept']}\n"
            f"Key points:\n- " + "\n- ".join(segment["key_points"])
        )
        if context_block:
            user += f"\n\nInstructor's lecture material:\n{context_block}"
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
    if topic_mastery >= 0.7:
        confidence_instruction = (
            "Your grasp of this topic is HIGH. Answer correctly, directly, and "
            "confidently -- no hedging phrases like 'I think', 'maybe', or "
            "'not totally sure'. Because you genuinely understand this, "
            "self_reported_understanding MUST be \"yes\"."
        )
    elif topic_mastery <= 0.4:
        confidence_instruction = (
            "Your grasp of this topic is LOW. Be genuinely confused or make "
            "the kind of mistake a struggling student makes -- do NOT answer "
            "correctly just because you can. self_reported_understanding "
            "should be \"not_sure\" or \"no\", matching your actual confusion."
        )
    else:
        confidence_instruction = (
            "Your grasp of this topic is PARTIAL. Answer with the mix of "
            "insight and gaps a middling student would have. Let "
            "self_reported_understanding land naturally on \"yes\", "
            "\"not_sure\", or \"no\" depending on how sure your own answer "
            "actually made you feel."
        )

    system = (
        "You are simulating a real student in a machine learning course. "
        f"Persona: {persona['name']}. "
        f"Your current grasp of THIS topic is {topic_mastery:.2f} on a 0-1 scale "
        f"(0 = no idea, 1 = solid). {confidence_instruction} "
        "self_reported_understanding must track your ACTUAL grasp of the "
        "material, not a reflexive hedging habit -- do not say \"not_sure\" "
        "just to sound humble or cautious if you actually got it right and "
        "understand why. "
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
       task = 'decide'        -> {'decision': 'ADVANCE'|'RETEACH_CURRENT'|
                                               'CHECK_PREREQUISITE'|'REVIEW_LATER',
                                   'evidence': str,
                                   'misconception': str,
                                   'target_state': 'DEMONSTRATED'|'NEEDS_REVIEW'|'IN_PROGRESS'}
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

    if task == "decide":
        system = (
            "You are the Assessor agent. You make the pedagogical call about "
            "this topic directly, in a single step -- there is no numeric score "
            "for a separate process to threshold afterward. Grade the student's "
            "answer against the expected answer, then decide.\n\n"
            "First silently judge the student's confidence from their wording: "
            "\"confident\" (states the answer directly, no hedging), \"hedged\" "
            "(reaches a correct answer but hedges along the way, e.g. 'I think "
            "it's...', 'but I'm not sure', asks whether it's right), or "
            "\"guessing\" (explicitly says they don't know or are just "
            "guessing). A correct-but-hedged or correct-but-guessing answer is "
            "NOT evidence of mastery -- never choose ADVANCE for an answer that "
            "only sounds lucky.\n\n"
            "Then choose exactly one decision:\n"
            "  ADVANCE            - correct AND confidently reasoned; the "
            "student has demonstrated mastery of this topic.\n"
            "  RETEACH_CURRENT    - wrong, hedged, or guessed, but the error "
            "looks like a misunderstanding of THIS topic specifically (not an "
            "upstream gap) -- a worked example on this topic should fix it.\n"
            "  CHECK_PREREQUISITE - the error pattern suggests the student is "
            "missing one of this topic's listed prerequisites, not the topic "
            "itself.\n"
            "  REVIEW_LATER       - weak or wrong, but doesn't cleanly point "
            "at this topic or a specific prerequisite; flag it without "
            "blocking progress.\n\n"
            "Return ONLY JSON with keys: "
            '"decision" (one of "ADVANCE", "RETEACH_CURRENT", '
            '"CHECK_PREREQUISITE", "REVIEW_LATER"), '
            '"evidence" (one sentence: what in the answer supports this decision), '
            '"misconception" (short description of the observed misconception, or "" if none), '
            '"target_state" (the new state for this topic: "DEMONSTRATED" if '
            'ADVANCE, "NEEDS_REVIEW" if REVIEW_LATER or CHECK_PREREQUISITE, '
            '"IN_PROGRESS" if RETEACH_CURRENT).'
        )
        user = (
            f"Topic: {kw['dimension']}\n"
            f"Prerequisites for this topic: {kw.get('prerequisite_dims') or 'none'}\n"
            f"Question: {kw['question']}\n"
            f"Expected answer: {kw['expected']}\n"
            f"Student answer: {kw['student_answer'].get('answer')}"
        )
        return call_json(system, user, max_tokens=300)

    raise ValueError(f"Unknown assessor task: {task}")

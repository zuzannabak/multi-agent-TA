"""
The four agents. Each is a single function that builds a prompt, calls the
model, and returns structured output. The orchestrator wires them together.

Contract summary:
  planner  : reads knowledge_state           -> decides teach / remediate / advance
  teacher  : given a segment + task          -> generates teaching text or worked example
  proposer : given the mini-unit just taught -> 0-3 concrete candidate questions
             (an empty list is a first-class, valid outcome)
  judge    : given the candidates            -> ASK (pick the single best one) or SKIP,
             no numeric score -- one direct call, same as the assessor
  student  : given content + question + mastery -> answers + self-reports understanding
  assessor : makes the ADVANCE / RETEACH_CURRENT / CHECK_PREREQUISITE /
             REVIEW_LATER call directly on a checked answer -> (orchestrator
             applies target_state to knowledge_state)

A concept only becomes DEMONSTRATED via a correct answer to a concrete
question the judge chose to ask -- never via self-report. The default is to
SKIP asking and keep teaching; asking requires the judge's justification.
"""
import json
import re

from llm import call_json
from retrieval import retrieve_for_segment, format_context


# ---------------------------------------------------------------------------
# Self-report ban-list
# ---------------------------------------------------------------------------
# "Do you understand?" style questions are worthless -- students say yes
# almost regardless of whether they actually understand. Any generated
# question matching one of these phrasings is rejected before it can be
# asked; the question proposer and judge both run this as a hard filter.
SELF_REPORT_PATTERNS = [
    r"do you understand",
    r"does (?:that|this) make sense",
    r"are you comfortable",
    r"are you familiar",
    r"do you feel",
    r"do you follow",
    r"is (?:that|this) clear",
    r"are you confident",
    r"do you get it",
]
_SELF_REPORT_RE = re.compile("|".join(SELF_REPORT_PATTERNS), re.IGNORECASE)


def is_self_report_question(question_text):
    """True if `question_text` asks the student to self-report understanding
    instead of demonstrating it via a concrete answer. Self-report questions
    are banned outright -- see SELF_REPORT_PATTERNS above."""
    return bool(_SELF_REPORT_RE.search(question_text))


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
# QUESTION PROPOSER
# ---------------------------------------------------------------------------
def propose_questions(segment, teaching_text):
    """Given the mini-unit just taught, propose 1-3 CONCRETE candidate
    questions -- or none, if nothing complete and assessable has been taught
    yet. An empty list is a first-class, valid outcome: returning nothing is
    better than asking a premature or weak question.

    Returns {'candidates': [{'question', 'expected', 'reveals_if_wrong'}, ...]}
    """
    system = (
        "You are the Question Proposer agent in a tutoring system. You just "
        "watched the Teacher teach a mini-unit. Propose 1-3 CONCRETE candidate "
        "questions that make the student DEMONSTRATE understanding through "
        "action: compute, choose, compare, classify, predict, explain, or "
        "find an error. NEVER propose a question that just asks the student "
        "to self-report understanding (e.g. 'do you understand', 'does that "
        "make sense') -- those are worthless and banned outright.\n\n"
        "Every candidate must be answerable using ONLY what the teaching text "
        "below actually covered -- do not require content that hasn't been "
        "taught yet. If the mini-unit only set up an idea and hasn't yet "
        "reached a complete, assessable concept, return an EMPTY candidates "
        "list. Returning nothing is better than asking a premature or weak "
        "question.\n\n"
        "Return ONLY JSON with key \"candidates\": a list (0 to 3 items) of "
        "objects, each with keys "
        '"question" (the concrete question), '
        '"expected" (the correct answer), '
        '"reveals_if_wrong" (one sentence: what a wrong answer would reveal '
        "about the student's misconception)."
    )
    user = (
        f"Concept: {segment['concept']}\n"
        f"Key points for this segment:\n- " + "\n- ".join(segment["key_points"]) +
        f"\n\nWhat the teacher just taught:\n{teaching_text}"
    )
    return call_json(system, user, max_tokens=500)


# ---------------------------------------------------------------------------
# QUESTION JUDGE
# ---------------------------------------------------------------------------
def judge_questions(candidates, segment):
    """Given the proposer's candidates, make ONE direct call: ASK (and pick
    the single best candidate) or SKIP (keep teaching, ask nothing). No
    numeric score or confidence threshold -- same style as the assessor's
    single-step decision. Also runs the self-report ban-list as a hard filter.

    Returns {'action': 'ASK'|'SKIP', 'reason': str, 'chosen': candidate|None}
    """
    clean = [c for c in candidates if not is_self_report_question(c["question"])]
    if not clean:
        return {
            "action": "SKIP",
            "reason": "No candidates (or none survived the self-report ban-list).",
            "chosen": None,
        }

    system = (
        "You are the Question Judge agent. You receive candidate questions "
        "proposed right after a mini-unit was taught. Decide ONE thing "
        "directly: ASK or SKIP. Do not produce a numeric score or confidence "
        "value -- make the call.\n\n"
        "Choose SKIP if none of the candidates are good enough: a candidate "
        "is NOT good enough if it only tests recall rather than real "
        "understanding, requires content that hasn't been taught yet, is "
        "confounded by a prerequisite gap instead of cleanly testing this "
        "concept, is ambiguous, or wouldn't change what the tutor does next "
        "regardless of the answer.\n\n"
        "Choose ASK if at least one candidate is good -- and pick the single "
        "best one.\n\n"
        "Return ONLY JSON with keys: "
        '"action" (one of "ASK", "SKIP"), '
        '"reason" (one sentence), '
        '"chosen_index" (0-based index into the candidate list if ASK, or '
        "null if SKIP)."
    )
    user = (
        f"Concept: {segment['concept']}\n\n"
        "Candidate questions:\n" +
        "\n".join(f"{i}. {c['question']} (expected: {c['expected']})"
                   for i, c in enumerate(clean))
    )
    result = call_json(system, user, max_tokens=300)

    if result.get("action") != "ASK" or result.get("chosen_index") is None:
        return {"action": "SKIP", "reason": result.get("reason", ""), "chosen": None}

    idx = result["chosen_index"]
    chosen = clean[idx] if isinstance(idx, int) and 0 <= idx < len(clean) else None
    if chosen is None or is_self_report_question(chosen["question"]):
        return {
            "action": "SKIP",
            "reason": "Judge's chosen question failed the ban-list check.",
            "chosen": None,
        }

    return {"action": "ASK", "reason": result.get("reason", ""), "chosen": chosen}


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
    """task = 'decide' -> {'decision': 'ADVANCE'|'RETEACH_CURRENT'|
                                        'CHECK_PREREQUISITE'|'REVIEW_LATER',
                            'evidence': str,
                            'misconception': str,
                            'target_state': 'DEMONSTRATED'|'NEEDS_REVIEW'|'IN_PROGRESS'}
    """
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

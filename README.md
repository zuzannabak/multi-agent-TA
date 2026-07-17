# Multi-Agent AI Teaching Assistant

A four-agent tutoring system for a graduate machine learning course (CIS 5526),
built as a research prototype. It teaches lecture material one segment at a time,
checks whether a student actually understands each concept, and adapts what it
teaches next based on where their understanding breaks down.

**Status:** work in progress.

## What it does

The system runs an interactive diagnostic loop for each lecture segment:

1. The **planner** reads the student's current knowledge state (a per-topic
   categorical mastery state, not a numeric score) and decides whether to
   teach, remediate a prerequisite, or advance.
2. The **teacher** teaches the segment — grounded in the actual course lecture via
   retrieval (RAG), not the model's generic knowledge.
3. The **student** (simulated, for testing) responds and self-reports understanding.
4. The **assessor** makes the pedagogical call directly in one step — ADVANCE,
   RETEACH_CURRENT, CHECK_PREREQUISITE, or REVIEW_LATER — with a one-sentence
   evidence note, and that decision sets the topic's new mastery state.

When a student is confused, the system doesn't just re-quiz — it serves a worked
example, then traces back through prerequisite dependencies to find where the
understanding actually breaks (e.g. a distance-metric confusion caused by a gap
in linear algebra), with a depth limit so a struggling student is never stuck.

## Key components

- **RAG over lecture material** — the merged lecture is chunked by section and
  embedded in a local ChromaDB vector store, so the teacher teaches using the
  instructor's own examples and phrasing.
- **Mastery graph** — each topic holds one categorical state (`NOT_SEEN`,
  `IN_PROGRESS`, `DEMONSTRATED`, `NEEDS_REVIEW`) plus an optional misconceptions
  list and evidence summary, not a 0-1 confidence score. A dependency map links
  each topic to its prerequisites so a weak showing can be traced upstream.
- **Single-step assessor decision** — the assessor makes the teach/reteach/
  check-prerequisite/advance call directly in one LLM call (ADVANCE,
  RETEACH_CURRENT, CHECK_PREREQUISITE, REVIEW_LATER), instead of emitting a
  numeric score for a separate step to threshold.
- **Model cost/quality comparison** — benchmarks LLM tiers (Gemini, GPT tiers) on
  grading accuracy vs. cost against a per-student budget.

## Files

| file | purpose |
|---|---|
| `orchestrator.py` | runs the diagnostic loop for a lecture segment |
| `agents.py` | the four agents: planner, teacher, student, assessor |
| `knowledge_state.py` | categorical mastery states, dependency map, segment definitions |
| `llm.py` | multi-provider LLM interface (Gemini / OpenAI) |
| `chunk_lecture.py` | chunks the merged lecture by section |
| `build_vectordb.py` | embeds chunks into local ChromaDB |
| `retrieval.py` | retrieves relevant lecture chunks per segment |
| `test_retrieval.py` | CLI tool to sanity-check retrieval quality |
| `compare_rag.py` | side-by-side demo of teaching with vs. without RAG |
| `model_comparison.py` | quality/cost benchmark across model tiers |
| `comparison_results.md` | saved results from a model-comparison run |

## Setup

```bash
pip install openai google-genai chromadb

# set whichever provider you're using
export OPENAI_API_KEY="your-key"      # default provider
# or
export GEMINI_API_KEY="your-key"
export LLM_PROVIDER="gemini"
```

## Running it

```bash
# one-time: build the vector store from the lecture
python chunk_lecture.py
python build_vectordb.py

# run the tutoring loop on a segment
python orchestrator.py

# compare teaching with vs. without lecture grounding
python compare_rag.py distance_metrics
```

## Architecture notes

The teacher and assessor are the two agents where output quality matters most:
the teacher needs to reflect the real lecture, and the assessor's decision writes
directly into the mastery graph that drives every planner decision — so a
mis-grade there is a correctness problem, not just a stylistic one.

State is represented, and the advance decision is made, deliberately without a
numeric confidence score: the assessor judges the student's answer (including
hedged or lucky-correct responses that shouldn't count as mastery) and lands on
one categorical decision — ADVANCE, RETEACH_CURRENT, CHECK_PREREQUISITE, or
REVIEW_LATER — in the same call, rather than emitting a score for a separate
step to threshold. The orchestrator applies the resulting state directly;
RETEACH_CURRENT retries once and CHECK_PREREQUISITE walks one level of the
dependency map, so a struggling student is never stuck on one segment.

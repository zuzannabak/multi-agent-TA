# Multi-Agent AI Teaching Assistant

A four-agent tutoring system for a graduate machine learning course (CIS 5526),
built as a research prototype. It teaches lecture material one segment at a time,
checks whether a student actually understands each concept, and adapts what it
teaches next based on where their understanding breaks down.

**Status:** work in progress.

## What it does

The system runs an interactive diagnostic loop for each lecture segment:

1. The **planner** reads the student's current knowledge state and decides whether
   to teach, remediate a prerequisite, or advance.
2. The **teacher** teaches the segment — grounded in the actual course lecture via
   retrieval (RAG), not the model's generic knowledge.
3. The **student** (simulated, for testing) responds and self-reports understanding.
4. The **assessor** evaluates the response, checks prerequisites when the student
   is unsure, and updates the knowledge-state vector that drives the next decision.

When a student is confused, the system doesn't just re-quiz — it serves a worked
example, then traces back through prerequisite dependencies to find where the
understanding actually breaks (e.g. a distance-metric confusion caused by a gap
in linear algebra), with a depth limit so a struggling student is never stuck.

## Key components

- **RAG over lecture material** — the merged lecture is chunked by section and
  embedded in a local ChromaDB vector store, so the teacher teaches using the
  instructor's own examples and phrasing.
- **Knowledge-state tracker** — a per-topic vector modeling what the student
  understands, plus a dependency map linking each topic to its prerequisites.
- **Model cost/quality comparison** — benchmarks LLM tiers (Gemini, GPT tiers) on
  grading accuracy vs. cost against a per-student budget.

## Files

| file | purpose |
|---|---|
| `orchestrator.py` | runs the diagnostic loop for a lecture segment |
| `agents.py` | the four agents: planner, teacher, student, assessor |
| `knowledge_state.py` | knowledge-state vector, dependency map, segment definitions |
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
the teacher needs to reflect the real lecture, and the assessor's score writes
directly into the knowledge-state vector that drives every planner decision — so
a mis-grade there is a correctness problem, not just a stylistic one.

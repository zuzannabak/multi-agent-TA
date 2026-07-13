"""
Side-by-side demo of what retrieval grounding buys the teacher agent: the
same segment taught once with the LLM's generic knowledge (use_rag=False)
and once grounded in the actual lecture (use_rag=True), plus the lecture
chunks that grounded the second version.

Run:
    python compare_rag.py                     # defaults to distance_metrics
    python compare_rag.py feature_scaling
"""
import sys
import textwrap

from knowledge_state import SEGMENTS
from agents import teacher
from retrieval import retrieve_for_segment

# Windows consoles often default stdout to a legacy codepage (e.g. cp1250)
# that can't render the em-dashes/curly quotes LLMs commonly output --
# force UTF-8 so the side-by-side columns don't get mangled in screenshots.
sys.stdout.reconfigure(encoding="utf-8")

COLUMN_WIDTH = 38
TOTAL_WIDTH = COLUMN_WIDTH * 2 + 3  # " | " separator


def wrap_to_lines(text, width):
    """Wrap text into a list of lines, preserving blank-line paragraph breaks."""
    lines = []
    for i, paragraph in enumerate(text.split("\n\n")):
        if i > 0:
            lines.append("")
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    return lines


def print_side_by_side(left_title, left_text, right_title, right_text):
    left_lines = wrap_to_lines(left_text, COLUMN_WIDTH)
    right_lines = wrap_to_lines(right_text, COLUMN_WIDTH)

    print(f"{left_title:<{COLUMN_WIDTH}} | {right_title}")
    print("-" * COLUMN_WIDTH + "-+-" + "-" * COLUMN_WIDTH)

    for i in range(max(len(left_lines), len(right_lines))):
        left = left_lines[i] if i < len(left_lines) else ""
        right = right_lines[i] if i < len(right_lines) else ""
        print(f"{left:<{COLUMN_WIDTH}} | {right}")


def print_header(text):
    print("=" * TOTAL_WIDTH)
    print(text.center(TOTAL_WIDTH))
    print("=" * TOTAL_WIDTH)


def main():
    dim = sys.argv[1] if len(sys.argv) > 1 else "distance_metrics"
    if dim not in SEGMENTS:
        print(f"No segment defined for '{dim}'. Available: {list(SEGMENTS)}")
        return
    segment = SEGMENTS[dim]

    print_header(f"RAG COMPARISON: {segment['dimension']}")
    print(f"Concept: {segment['concept']}")
    print()

    print("Generating without RAG (generic knowledge)...")
    without_rag = teacher(segment, "teach", use_rag=False)["teaching_text"]

    print("Generating with RAG (grounded in lecture)...")
    with_rag = teacher(segment, "teach", use_rag=True)["teaching_text"]
    print()

    print_side_by_side(
        "WITHOUT RAG (generic knowledge)", without_rag,
        "WITH RAG (grounded in lecture)", with_rag,
    )

    print()
    print_header("RETRIEVED LECTURE CHUNKS (grounded the RAG version)")
    chunks = retrieve_for_segment(segment)
    if not chunks:
        print("(none found for this dimension)")
    for i, c in enumerate(chunks, start=1):
        preview = c["chunk_text"].replace("\n", " ")
        if len(preview) > 150:
            preview = preview[:150] + "..."
        print(f"{i}. [{c['chunk_id']}] {c['section_title']}")
        print(f"   {preview}\n")


if __name__ == "__main__":
    main()

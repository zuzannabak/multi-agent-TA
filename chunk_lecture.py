"""
Chunks the merged KNN lecture notes by section (## headers) so the teacher
agent can eventually be grounded in the actual lecture instead of the LLM's
generic knowledge of KNN.

Chunking rule: split on ## headers. If a section's estimated token count
exceeds SECTION_TOKEN_LIMIT, split it further on its ### subheaders.

Token counts are estimated with a chars/4 heuristic (no tokenizer dependency
installed in this project) -- good enough for a chunking-size decision.

Run:
    python chunk_lecture.py
"""
import json
import re

SOURCE_FILE = "merged_knn_lecture_notes.md"
OUTPUT_FILE = "chunks.json"
SECTION_TOKEN_LIMIT = 800

KNOWLEDGE_DIMENSIONS = [
    "supervised_learning_setup",
    "model_as_algorithm",
    "classification_regression",
    "knn_algorithm",
    "distance_metrics",
    "feature_scaling",
    "train_test_eval",
    "cost_sensitive_eval",
    "choosing_k_dimensionality",
]

# Maps each ## section number (in document order) to the knowledge dimension
# it primarily teaches. Hand-mapped by reading the section content once --
# update this if sections are added/reordered/split in the source document.
SECTION_DIMENSION_MAP = {
    1: "supervised_learning_setup",   # Transition into Supervised Learning
    2: "model_as_algorithm",          # Three Core Questions (what is f?)
    3: "classification_regression",   # Mental Pictures: classification/regression
    4: "knn_algorithm",               # Discovering nearest-neighbor reasoning
    5: "knn_algorithm",               # Formal setup of KNN
    6: "knn_algorithm",               # KNN pseudocode and computational cost
    7: "distance_metrics",            # Hyperparameters: k and distance
    8: "distance_metrics",            # Distance functions (Euclidean, L-p, Hamming...)
    9: "feature_scaling",             # Feature scaling and preprocessing
    10: "train_test_eval",            # Train/test split
    11: "train_test_eval",            # Confusion table
    12: "train_test_eval",            # Accuracy, precision, recall, F1
    13: "cost_sensitive_eval",        # Cost of mistakes
    14: "train_test_eval",            # Multiclass classification (evaluation extension)
    15: "choosing_k_dimensionality",  # Live demo: effect of k on decision regions
    16: "train_test_eval",            # Live demo: train/test split + accuracy on Iris
    17: "knn_algorithm",              # Live demo: KNN vs. other classifiers
    18: "knn_algorithm",              # Is KNN a good algorithm? (dense-neighborhood need)
    19: "choosing_k_dimensionality",  # Curse of dimensionality
    20: "knn_algorithm",              # Consolidated takeaways
}

SECTION_HEADER_RE = re.compile(r"^## (\d+)\.\s*(.+)$", re.MULTILINE)
SUBSECTION_HEADER_RE = re.compile(r"^### (.+)$", re.MULTILINE)


def estimate_tokens(text):
    return len(text) // 4


def clean(text):
    # Sections end with a trailing "---" horizontal rule; drop it.
    return re.sub(r"\n---\s*$", "", text).strip()


def split_into_sections(document_text):
    """Split the document on ## N. headers. Returns a list of
    (section_number, section_title, section_body) tuples in document order."""
    matches = list(SECTION_HEADER_RE.finditer(document_text))
    sections = []
    for i, m in enumerate(matches):
        number = int(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(document_text)
        body = clean(document_text[start:end])
        sections.append((number, title, body))
    return sections


def split_into_subsections(section_body):
    """Split a section's body on its ### subheaders. Returns a list of
    (subheader_title_or_None, subsection_text) tuples. If there are no ###
    headers, returns [(None, section_body)]."""
    matches = list(SUBSECTION_HEADER_RE.finditer(section_body))
    if not matches:
        return [(None, section_body)]

    subsections = []
    # Text before the first ### header (rare, but keep it if present).
    preamble = section_body[: matches[0].start()].strip()
    if preamble:
        subsections.append((None, preamble))

    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(section_body)
        text = section_body[start:end].strip()
        subsections.append((title, text))
    return subsections


def build_chunks(sections):
    chunks = []
    for number, title, body in sections:
        dimension = SECTION_DIMENSION_MAP.get(number)
        if dimension is None:
            raise ValueError(f"No dimension mapping for section {number}: {title}")

        full_text = f"## {number}. {title}\n\n{body}"
        if estimate_tokens(full_text) <= SECTION_TOKEN_LIMIT:
            chunks.append({
                "chunk_id": f"{number}",
                "section_title": title,
                "segment_dimension": dimension,
                "chunk_text": full_text,
                "approx_tokens": estimate_tokens(full_text),
            })
            continue

        # Oversized section: split further on ### subheaders.
        for j, (subtitle, subtext) in enumerate(split_into_subsections(body), start=1):
            heading = f"## {number}. {title}"
            if subtitle:
                heading += f" -- {subtitle}"
            sub_full_text = f"{heading}\n\n{subtext}"
            chunks.append({
                "chunk_id": f"{number}.{j}",
                "section_title": f"{title} -- {subtitle}" if subtitle else title,
                "segment_dimension": dimension,
                "chunk_text": sub_full_text,
                "approx_tokens": estimate_tokens(sub_full_text),
            })

    return chunks


def print_summary(chunks):
    print(f"Total chunks: {len(chunks)}\n")

    counts = {dim: 0 for dim in KNOWLEDGE_DIMENSIONS}
    for c in chunks:
        counts[c["segment_dimension"]] += 1

    print("Chunks per dimension:")
    for dim in KNOWLEDGE_DIMENSIONS:
        n = counts[dim]
        flag = "  <-- NO COVERAGE" if n == 0 else ""
        print(f"  {dim:<28} {n}{flag}")


if __name__ == "__main__":
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        document_text = f.read()

    sections = split_into_sections(document_text)
    chunks = build_chunks(sections)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print_summary(chunks)
    print(f"\nWrote {len(chunks)} chunks to {OUTPUT_FILE}")

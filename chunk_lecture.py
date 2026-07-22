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

SOURCE_FILE = "merged_knn_lecture_clean.md"
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
#
# Remapped 2026-07-22 for merged_knn_lecture_clean.md, which strips non-lecture
# content out of the merged_knn_lecture_new_07222026.md revision: the document
# header/source-hierarchy note, old section 1 ("why the instructor writes by
# hand" -- pedagogy, no ML content), the "### Source status" asides inside
# what were sections 8/9/10, old section 23 ("end of lecture" -- logistics),
# and the trailing "Consolidated lecture flow" summary (a one-line-per-topic
# recap that matched almost any retrieval query without teaching anything).
# Removing old section 1 shifted every remaining section down by one and
# renumbered 2-22 -> 1-21; the dimension assignments themselves are otherwise
# unchanged from the prior mapping.
SECTION_DIMENSION_MAP = {
    1: "supervised_learning_setup",    # The dataset: examples, features, labels
    2: "supervised_learning_setup",    # Where the dataset comes from: sampling the world
    3: "supervised_learning_setup",    # The supervised-learning loop and prediction model f
    4: "model_as_algorithm",           # Three organizing questions (what/how good/how learn is f)
    5: "classification_regression",    # Classification vs. regression: mental picture
    6: "knn_algorithm",                # Discovering nearest-neighbor reasoning
    7: "knn_algorithm",                # Formal KNN setup and prediction rule
    8: "knn_algorithm",                # Euclidean distance, pseudocode, computational cost
    9: "choosing_k_dimensionality",    # Hyperparameters and the problem of choosing k
    10: "distance_metrics",            # Distance measures (L-p, Manhattan, cosine, Hamming)
    11: "feature_scaling",             # Feature scaling and normalization
    12: "train_test_eval",             # Evaluating KNN: split the data first
    13: "train_test_eval",             # Confusion matrix
    14: "train_test_eval",             # Accuracy, error rate, precision, recall, F1
    15: "cost_sensitive_eval",         # Unequal costs of mistakes
    16: "train_test_eval",             # Extending evaluation to multiclass classification
    17: "choosing_k_dimensionality",   # Iris demo: KNN decision regions across k
    18: "choosing_k_dimensionality",   # Selecting k through repeated train-test experiments
    19: "knn_algorithm",               # Demonstration of other classifier families
    20: "knn_algorithm",               # Is KNN a good algorithm? (dense-neighborhood need)
    21: "choosing_k_dimensionality",   # Curse of dimensionality
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

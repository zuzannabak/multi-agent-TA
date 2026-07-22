"""
Shared state for the tutoring agent loop.

This is the spine of the system:
  - the ASSESSOR writes to knowledge_state (sets a new categorical state after
    each check, via a single-step ADVANCE/RETEACH/CHECK_PREREQUISITE/REVIEW
    decision)
  - the PLANNER reads from it (decides teach vs. remediate vs. advance)

Each dimension is a categorical mastery state, not a numeric score:
    NOT_SEEN      -> not yet taught
    IN_PROGRESS   -> taught, not yet confirmed either way
    DEMONSTRATED  -> student showed correct, confident understanding
    NEEDS_REVIEW  -> covered, but evidence was weak/hedged/wrong; flagged for
                     a later pass instead of blocking progress now

v2 (2026-07-22): the old 9 lecture-topic + 5 prerequisite dimensions
conflated "knowing" and "doing" into one axis per topic. Split into 16
dimensions across 4 groups so a student can be strong on one axis and weak
on the other for the same idea -- e.g. grasping "guilt by association"
conceptually while still fumbling the arithmetic of a distance formula:
    A - foundational skills   (seeded from the pretest)
    B - reading representations
    C - conceptual understanding
    D - technical execution
"""

# ---------------------------------------------------------------------------
# Mastery states
# ---------------------------------------------------------------------------
NOT_SEEN = "NOT_SEEN"
IN_PROGRESS = "IN_PROGRESS"
DEMONSTRATED = "DEMONSTRATED"
NEEDS_REVIEW = "NEEDS_REVIEW"

MASTERY_STATES = {NOT_SEEN, IN_PROGRESS, DEMONSTRATED, NEEDS_REVIEW}


# ---------------------------------------------------------------------------
# Dimension groups
# ---------------------------------------------------------------------------
GROUP_A = [
    "linear_algebra",          # vectors, coordinates, norms
    "formula_application",     # can substitute numbers into a given formula
    "probability_stats",       # proportions, distributions
    "computational_thinking",  # big-O, cost of an algorithm
    "python_reading",
]

GROUP_B = [
    "reading_scatter_plots",     # what a point on a plot means
    "reading_decision_regions",  # what the background colour means
]

GROUP_C = [
    "supervised_learning_framing",  # D, features, labels, sampling from the world
    "model_as_algorithm",           # a model can be a procedure, not just a formula
    "knn_intuition",                # guilt by association, why neighbours vote
    "hyperparameter_intuition",     # why very small / very large k hurts
    "evaluation_reasoning",         # why hold out test data, why accuracy isn't enough
    "dimensionality_intuition",     # why neighbourhoods disappear in high dimensions
]

GROUP_D = [
    "distance_computation",  # compute Euclidean / Manhattan / Hamming by hand
    "scaling_computation",   # apply min-max and standardization
    "metric_computation",    # compute precision / recall / F1 / cost from a confusion matrix
]

ALL_DIMENSIONS = GROUP_A + GROUP_B + GROUP_C + GROUP_D

DIMENSION_GROUPS = {
    **{d: "A" for d in GROUP_A},
    **{d: "B" for d in GROUP_B},
    **{d: "C" for d in GROUP_C},
    **{d: "D" for d in GROUP_D},
}


def _topic_state(state, group, misconceptions=None, evidence_summary=""):
    return {
        "state": state,
        "group": group,
        "misconceptions": misconceptions or [],
        "evidence_summary": evidence_summary,
    }


def primary_conceptual_dimension(dimensions):
    """Given a segment's `dimensions` list, return its Group C (conceptual)
    entry. Retrieval and today's single-dimension state tracking both key off
    of this one dimension until the assessor is reworked (next step) to
    update a segment's conceptual, technical, and foundational dimensions
    independently from one check."""
    for d in dimensions:
        if DIMENSION_GROUPS.get(d) == "C":
            return d
    raise ValueError(f"No Group C (conceptual) dimension found in {dimensions}")


# ---------------------------------------------------------------------------
# Knowledge-state vector
# ---------------------------------------------------------------------------
# Group A dims are set once from the pretest (here: hardcoded per persona).
# Groups B, C, D start NOT_SEEN and are updated live during the lecture.

def fresh_knowledge_state(prereq_scores, prereq_threshold=0.7):
    """Build a starting knowledge_state from a persona's pretest scores.

    prereq_scores: {dimension: 0.0-1.0} pretest results, one entry per
        Group A dimension (linear_algebra, formula_application,
        probability_stats, computational_thinking, python_reading).
    prereq_threshold: pretest score at/above which a Group A dimension starts
        DEMONSTRATED; below it, it starts NEEDS_REVIEW so the planner can
        flag it for remediation before it blocks a segment.
    """
    def prereq_state(score):
        return DEMONSTRATED if score >= prereq_threshold else NEEDS_REVIEW

    knowledge_state = {
        dim: _topic_state(prereq_state(prereq_scores[dim]), "A")
        for dim in GROUP_A
    }
    for dim in GROUP_B + GROUP_C + GROUP_D:
        knowledge_state[dim] = _topic_state(NOT_SEEN, DIMENSION_GROUPS[dim])
    return knowledge_state


# ---------------------------------------------------------------------------
# Dependency map — consulted by the assessor to decide if a weak showing
# is actually caused by an upstream gap.
# ---------------------------------------------------------------------------
DEPENDENCY_MAP = {
    # Group D — technical execution
    "distance_computation": ["linear_algebra"],
    "scaling_computation":  ["distance_computation", "formula_application"],
    "metric_computation":   ["probability_stats"],

    # Group C — conceptual understanding
    "supervised_learning_framing": [],
    "model_as_algorithm":          [],
    "knn_intuition":               ["supervised_learning_framing", "model_as_algorithm"],
    "hyperparameter_intuition":    ["knn_intuition", "reading_decision_regions"],
    "evaluation_reasoning":        ["probability_stats", "supervised_learning_framing"],
    "dimensionality_intuition":    ["knn_intuition", "probability_stats"],

    # Groups A/B — foundational; no ML prerequisites of their own
    "linear_algebra": [],
    "formula_application": [],
    "probability_stats": [],
    "computational_thinking": [],
    "python_reading": [],
    "reading_scatter_plots": [],
    "reading_decision_regions": [],
}


# ---------------------------------------------------------------------------
# Segment definitions.
# One is fully specified (distance_metrics) to run end-to-end.
# The others follow the SAME schema — paste your ChatGPT branching output
# into this structure to add them.
#
# Schema:
#   dimensions         : list of knowledge_state keys this segment can supply
#                         evidence for -- STUDENT KNOWLEDGE, not lecture
#                         content. Usually one conceptual (Group C) dim, one
#                         technical (Group D) dim, and sometimes one
#                         foundational (Group A) dim.
#   lecture_topic      : the chunk_lecture.py / vector-store tag (old 9-topic
#                         scheme) this segment's LECTURE CONTENT should
#                         retrieve from. A separate taxonomy from `dimensions`
#                         on purpose -- retrieval.py filters on this field.
#   concept            : one-line description (fed to the teacher)
#   key_points         : bullets the teacher must cover (teacher GENERATES from these)
#   gate_question      : the "do you understand X?" check (authored, delivered by teacher)
#   confirmation       : {question, expected}  (authored, graded by assessor)
#   prerequisite_diagnostics : [{dimension, question, expected}]  (authored, graded by assessor)
# ---------------------------------------------------------------------------
SEGMENTS = {
    "distance_metrics": {
        "dimensions": ["knn_intuition", "distance_computation", "linear_algebra"],
        "lecture_topic": "distance_metrics",
        "concept": "Why KNN needs a distance metric, and why different metrics define 'near' differently.",
        "key_points": [
            "KNN's whole prediction depends on which points count as 'nearest'.",
            "Euclidean = straight-line distance; Manhattan = grid-walk distance.",
            "For binary vectors, Hamming distance counts disagreeing positions.",
            "The same two points can be close under one metric and far under another.",
            "Choosing a metric is a modeling decision based on the data, not automatic.",
        ],
        "gate_question": "Do you understand why KNN needs a distance metric, and why different metrics can define 'near' differently?",
        "confirmation": {
            "question": "What is the Manhattan distance between the points (1, 2) and (4, 6)?",
            "expected": "7  (|1-4| + |2-6| = 3 + 4 = 7)",
        },
        "prerequisite_diagnostics": [
            {
                "dimension": "linear_algebra",
                "question": "What is the coordinate-wise difference between the vectors (4, 6) and (1, 2)?",
                "expected": "(3, 4)",
            },
        ],
    },

    "feature_scaling": {
        "dimensions": ["knn_intuition", "scaling_computation", "formula_application"],
        "lecture_topic": "feature_scaling",
        "concept": "Why unscaled features distort KNN's distance calculations, and how min-max scaling or standardization fixes it.",
        "key_points": [
            "Distance-based methods like KNN are sensitive to each feature's numeric scale.",
            "A feature with a much larger range dominates the distance calculation, regardless of how important it actually is.",
            "Example: CPU size in mm (~1e-3) vs. processor speed in GHz (~1e9) -- speed alone would decide 'nearness'.",
            "Min-max scaling maps a feature to [0,1]: x' = (x - min(x)) / (max(x) - min(x)).",
            "Standardization maps a feature to mean 0, std 1: x' = (x - mean(x)) / std(x).",
            "The goal is to give every feature an equal chance to influence distance, not let scale alone decide importance.",
        ],
        "gate_question": "Do you understand why unscaled features can distort KNN's distance calculations, and how feature scaling fixes it?",
        "confirmation": {
            "question": "A feature x1 ranges from min=2 to max=10 across the dataset. Using min-max scaling, what is the scaled value of x1=8?",
            "expected": "0.75  ((8-2)/(10-2) = 6/8 = 0.75)",
        },
        "prerequisite_diagnostics": [
            {
                "dimension": "distance_computation",
                "question": "Why would a feature measured in GHz (values near 10^9) dominate a Euclidean distance calculation compared to a feature measured in millimeters (values near 10^-3)?",
                "expected": "Euclidean distance sums squared differences across features; a feature with a much larger numeric scale produces much larger differences, so it dominates the total distance regardless of the other feature's actual importance.",
            },
        ],
    },

    # --- add point 1, 2, ... here using the same schema ---
    # "supervised_learning_framing": { ... },
}

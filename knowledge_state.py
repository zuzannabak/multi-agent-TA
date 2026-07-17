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
"""

# ---------------------------------------------------------------------------
# Mastery states
# ---------------------------------------------------------------------------
NOT_SEEN = "NOT_SEEN"
IN_PROGRESS = "IN_PROGRESS"
DEMONSTRATED = "DEMONSTRATED"
NEEDS_REVIEW = "NEEDS_REVIEW"

MASTERY_STATES = {NOT_SEEN, IN_PROGRESS, DEMONSTRATED, NEEDS_REVIEW}


def _topic_state(state, misconceptions=None, evidence_summary=""):
    return {
        "state": state,
        "misconceptions": misconceptions or [],
        "evidence_summary": evidence_summary,
    }


# ---------------------------------------------------------------------------
# Knowledge-state vector
# ---------------------------------------------------------------------------
# Prerequisite dims are set once from the pretest (here: hardcoded per persona).
# Lecture-topic dims start NOT_SEEN and are updated live during the lecture.

def fresh_knowledge_state(prereq_scores, prereq_threshold=0.7):
    """Build a starting knowledge_state from a persona's pretest scores.

    prereq_scores: {dimension: 0.0-1.0} pretest results.
    prereq_threshold: pretest score at/above which a prerequisite starts
        DEMONSTRATED; below it, the prerequisite starts NEEDS_REVIEW so the
        planner can flag it for remediation before it blocks a segment.
    """
    def prereq_state(score):
        return DEMONSTRATED if score >= prereq_threshold else NEEDS_REVIEW

    return {
        # --- prerequisites (from pretest) ---
        "linear_algebra":     _topic_state(prereq_state(prereq_scores["linear_algebra"])),
        "calculus":           _topic_state(prereq_state(prereq_scores["calculus"])),
        "probability_stats":  _topic_state(prereq_state(prereq_scores["probability_stats"])),
        "big_o_analysis":     _topic_state(prereq_state(prereq_scores["big_o_analysis"])),
        "python":             _topic_state(prereq_state(prereq_scores["python"])),
        # --- lecture topics (not yet taught) ---
        "supervised_learning_setup": _topic_state(NOT_SEEN),
        "model_as_algorithm":        _topic_state(NOT_SEEN),
        "classification_regression": _topic_state(NOT_SEEN),
        "knn_algorithm":             _topic_state(NOT_SEEN),
        "distance_metrics":          _topic_state(NOT_SEEN),
        "feature_scaling":           _topic_state(NOT_SEEN),
        "train_test_eval":           _topic_state(NOT_SEEN),
        "cost_sensitive_eval":       _topic_state(NOT_SEEN),
        "choosing_k_dimensionality": _topic_state(NOT_SEEN),
    }


# ---------------------------------------------------------------------------
# Dependency map — consulted by the assessor to decide if a weak showing
# is actually caused by an upstream gap.
# ---------------------------------------------------------------------------
DEPENDENCY_MAP = {
    "distance_metrics":          ["linear_algebra"],
    "feature_scaling":           ["distance_metrics"],
    "knn_algorithm":             ["distance_metrics", "model_as_algorithm"],
    "train_test_eval":           ["probability_stats"],
    "cost_sensitive_eval":       ["train_test_eval"],
    "choosing_k_dimensionality": ["distance_metrics", "probability_stats"],
    # points 1-3 have no ML prerequisites (only general pretest background)
    "supervised_learning_setup": [],
    "model_as_algorithm":        [],
    "classification_regression": [],
}


# ---------------------------------------------------------------------------
# Segment definitions.
# One is fully specified (distance_metrics) to run end-to-end.
# The others follow the SAME schema — paste your ChatGPT branching output
# into this structure to add them.
#
# Schema:
#   dimension  : which knowledge_state key this segment updates
#   concept    : one-line description (fed to the teacher)
#   key_points : bullets the teacher must cover (teacher GENERATES from these)
#
# There is no authored confirmation question or prerequisite-diagnostic Q&A
# anymore -- the proposer generates candidate questions live from whatever
# the teacher actually taught, and the judge picks one (or skips). See
# agents.py: propose_questions / judge_questions.
# ---------------------------------------------------------------------------
SEGMENTS = {
    "distance_metrics": {
        "dimension": "distance_metrics",
        "concept": "Why KNN needs a distance metric, and why different metrics define 'near' differently.",
        "key_points": [
            "KNN's whole prediction depends on which points count as 'nearest'.",
            "Euclidean = straight-line distance; Manhattan = grid-walk distance.",
            "For binary vectors, Hamming distance counts disagreeing positions.",
            "The same two points can be close under one metric and far under another.",
            "Choosing a metric is a modeling decision based on the data, not automatic.",
        ],
    },

    "feature_scaling": {
        "dimension": "feature_scaling",
        "concept": "Why unscaled features distort KNN's distance calculations, and how min-max scaling or standardization fixes it.",
        "key_points": [
            "Distance-based methods like KNN are sensitive to each feature's numeric scale.",
            "A feature with a much larger range dominates the distance calculation, regardless of how important it actually is.",
            "Example: CPU size in mm (~1e-3) vs. processor speed in GHz (~1e9) -- speed alone would decide 'nearness'.",
            "Min-max scaling maps a feature to [0,1]: x' = (x - min(x)) / (max(x) - min(x)).",
            "Standardization maps a feature to mean 0, std 1: x' = (x - mean(x)) / std(x).",
            "The goal is to give every feature an equal chance to influence distance, not let scale alone decide importance.",
        ],
    },

    # --- add point 1, 2, ... here using the same schema ---
    # "supervised_learning_setup": { ... },
}

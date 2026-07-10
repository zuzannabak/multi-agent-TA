"""
Shared state for the tutoring agent loop.

This is the spine of the system:
  - the ASSESSOR writes to knowledge_state (updates scores after each check)
  - the PLANNER reads from it (decides advance vs. remediate)

Each dimension: {"score": 0.0-1.0, "taught": bool}
  taught=False -> not yet covered (distinct from "covered but failed" = taught=True, low score)
"""

# ---------------------------------------------------------------------------
# Knowledge-state vector
# ---------------------------------------------------------------------------
# Prerequisite dims are set once from the pretest (here: hardcoded per persona).
# Lecture-topic dims start taught=False and are updated live during the lecture.

def fresh_knowledge_state(prereq_scores):
    """Build a starting knowledge_state from a persona's pretest scores."""
    return {
        # --- prerequisites (from pretest) ---
        "linear_algebra":     {"score": prereq_scores["linear_algebra"],  "taught": True},
        "calculus":           {"score": prereq_scores["calculus"],        "taught": True},
        "probability_stats":  {"score": prereq_scores["probability_stats"], "taught": True},
        "big_o_analysis":     {"score": prereq_scores["big_o_analysis"],   "taught": True},
        "python":             {"score": prereq_scores["python"],          "taught": True},
        # --- lecture topics (not yet taught) ---
        "supervised_learning_setup": {"score": 0.0, "taught": False},
        "model_as_algorithm":        {"score": 0.0, "taught": False},
        "classification_regression": {"score": 0.0, "taught": False},
        "knn_algorithm":             {"score": 0.0, "taught": False},
        "distance_metrics":          {"score": 0.0, "taught": False},
        "feature_scaling":           {"score": 0.0, "taught": False},
        "train_test_eval":           {"score": 0.0, "taught": False},
        "cost_sensitive_eval":       {"score": 0.0, "taught": False},
        "choosing_k_dimensionality": {"score": 0.0, "taught": False},
    }


# ---------------------------------------------------------------------------
# Dependency map — consulted by the assessor to decide if a low score
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
#   dimension          : which knowledge_state key this segment updates
#   concept            : one-line description (fed to the teacher)
#   key_points         : bullets the teacher must cover (teacher GENERATES from these)
#   gate_question      : the "do you understand X?" check (authored, delivered by teacher)
#   confirmation       : {question, expected}  (authored, graded by assessor)
#   prerequisite_diagnostics : [{dimension, question, expected}]  (authored, graded by assessor)
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

    # --- add point 1, 2, ... here using the same schema ---
    # "supervised_learning_setup": { ... },
}

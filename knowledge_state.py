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
        "gate_question": "Do you understand why unscaled features can distort KNN's distance calculations, and how feature scaling fixes it?",
        "confirmation": {
            "question": "A feature x1 ranges from min=2 to max=10 across the dataset. Using min-max scaling, what is the scaled value of x1=8?",
            "expected": "0.75  ((8-2)/(10-2) = 6/8 = 0.75)",
        },
        "prerequisite_diagnostics": [
            {
                "dimension": "distance_metrics",
                "question": "Why would a feature measured in GHz (values near 10^9) dominate a Euclidean distance calculation compared to a feature measured in millimeters (values near 10^-3)?",
                "expected": "Euclidean distance sums squared differences across features; a feature with a much larger numeric scale produces much larger differences, so it dominates the total distance regardless of the other feature's actual importance.",
            },
        ],
    },

    # --- add point 1, 2, ... here using the same schema ---
    # "supervised_learning_setup": { ... },
}

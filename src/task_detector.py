"""
TaskDetector
============
6-rule heuristic that detects classification vs regression
and returns a confidence score (0–100%).

Rules (evaluated in order, each adds weight):
  1. dtype is bool                          → strong classification signal
  2. dtype is object / category / string    → strong classification signal
  3. nunique ≤ 2                            → binary classification
  4. nunique ≤ CLASSIFICATION_THRESHOLD     → multi-class classification
  5. nunique / len > HIGH_CARDINALITY_RATIO → strong regression signal
  6. All values are integers                → weak regression signal
     (unless already classified by rules 1-4)

Confidence is the weighted vote share of the winning side (50–100%).
"""

import pandas as pd
import numpy as np

# Thresholds (can be overridden via Config if needed)
CLASSIFICATION_THRESHOLD = 20
HIGH_CARDINALITY_RATIO   = 0.05   # if nunique/n > this → regression signal


class TaskDetectionResult:
    """Holds the detection outcome + explanation."""

    def __init__(self, task: str, confidence: float, rules_fired: list[str]):
        self.task        = task           # "classification" | "regression"
        self.confidence  = confidence     # 0.0 – 1.0
        self.rules_fired = rules_fired    # human-readable list

    @property
    def confidence_pct(self) -> str:
        return f"{self.confidence * 100:.0f}%"

    def __repr__(self) -> str:
        return (
            f"TaskDetectionResult(task='{self.task}', "
            f"confidence={self.confidence_pct}, "
            f"rules={self.rules_fired})"
        )

    def summary(self) -> str:
        lines = [
            f"  Detected task : {self.task.upper()}",
            f"  Confidence    : {self.confidence_pct}",
            f"  Rules fired   :",
        ]
        for r in self.rules_fired:
            lines.append(f"    • {r}")
        return "\n".join(lines)


class TaskDetector:
    """
    Stateless helper — call detect(y) to get a TaskDetectionResult.

    Parameters
    ----------
    forced_task : None | "classification" | "regression"
        If set, skips heuristics and returns the forced task at 100% confidence.
    """

    # Rule weights  (classification_weight, regression_weight)
    _RULES = {
        "dtype_bool":         (3, 0),
        "dtype_object":       (3, 0),
        "binary_target":      (3, 0),
        "low_cardinality":    (2, 0),
        "high_cardinality":   (0, 3),
        "integer_values":     (0, 1),
        "float_values":       (0, 2),
    }

    def __init__(self, forced_task: str | None = None):
        if forced_task not in (None, "auto", "classification", "regression"):
            raise ValueError(f"forced_task must be None/auto/classification/regression, got '{forced_task}'")
        self.forced_task = forced_task if forced_task != "auto" else None

    def detect(self, y: pd.Series) -> TaskDetectionResult:
        """Run heuristics and return TaskDetectionResult."""

        # Forced override
        if self.forced_task and self.forced_task != "auto":
            return TaskDetectionResult(
                task        = self.forced_task,
                confidence  = 1.0,
                rules_fired = [f"Forced by user: {self.forced_task}"],
            )

        clf_score  = 0
        reg_score  = 0
        rules_fired = []

        n_unique = y.nunique()
        n_total  = len(y)
        dtype_str = str(y.dtype)

        # Rule 1 — bool dtype
        if y.dtype == bool or dtype_str == "bool":
            w = self._RULES["dtype_bool"]
            clf_score += w[0]
            rules_fired.append(f"Rule 1: dtype is bool → classification (+{w[0]})")

        # Rule 2 — object / category / string
        elif dtype_str in ("object", "category") or "string" in dtype_str:
            w = self._RULES["dtype_object"]
            clf_score += w[0]
            rules_fired.append(f"Rule 2: dtype is {dtype_str} → classification (+{w[0]})")

        else:
            # Rule 3 — binary numeric (0/1 or 2 unique values)
            if n_unique <= 2:
                w = self._RULES["binary_target"]
                clf_score += w[0]
                rules_fired.append(f"Rule 3: only {n_unique} unique values → classification (+{w[0]})")

            # Rule 4 — low cardinality
            elif n_unique <= CLASSIFICATION_THRESHOLD:
                w = self._RULES["low_cardinality"]
                clf_score += w[0]
                rules_fired.append(
                    f"Rule 4: {n_unique} unique values ≤ {CLASSIFICATION_THRESHOLD} → classification (+{w[0]})"
                )

            # Rule 5 — high cardinality ratio
            if n_unique / n_total > HIGH_CARDINALITY_RATIO and n_unique > CLASSIFICATION_THRESHOLD:
                w = self._RULES["high_cardinality"]
                reg_score += w[1]
                rules_fired.append(
                    f"Rule 5: {n_unique}/{n_total} unique ratio > {HIGH_CARDINALITY_RATIO} → regression (+{w[1]})"
                )

            # Rule 6a — float values
            if np.issubdtype(y.dtype, np.floating):
                w = self._RULES["float_values"]
                reg_score += w[1]
                rules_fired.append(f"Rule 6a: float dtype → regression (+{w[1]})")

            # Rule 6b — integer values with high cardinality
            elif np.issubdtype(y.dtype, np.integer) and n_unique > CLASSIFICATION_THRESHOLD:
                w = self._RULES["integer_values"]
                reg_score += w[1]
                rules_fired.append(f"Rule 6b: integer dtype, high cardinality → regression (+{w[1]})")

        # Determine winner + confidence
        total = clf_score + reg_score
        if total == 0:
            # Fallback: no rules fired → default to classification
            task       = "classification"
            confidence = 0.5
            rules_fired.append("Fallback: no strong signal → defaulting to classification")
        elif clf_score >= reg_score:
            task       = "classification"
            confidence = clf_score / total
        else:
            task       = "regression"
            confidence = reg_score / total

        # Clamp confidence to [0.5, 1.0] — we're always at least 50%
        confidence = max(0.5, confidence)

        return TaskDetectionResult(
            task        = task,
            confidence  = round(confidence, 2),
            rules_fired = rules_fired,
        )

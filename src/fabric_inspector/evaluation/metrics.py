"""Defect-class metrics and validation-only threshold selection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def metrics(labels, predictions, scores=None) -> dict:
    """Use positive class 1 and zero for undefined precision/recall.

    Prediction -1 is an unresolved VLM output: it counts as an error for its
    true class and remains in every denominator. It is not a ranking score.
    """
    y, p = np.asarray(labels, dtype=int), np.asarray(predictions, dtype=int)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Nonempty aligned one-dimensional labels/predictions required")
    if not np.isin(y, [0, 1]).all() or not np.isin(p, [-1, 0, 1]).all():
        raise ValueError("Labels must be binary; predictions may additionally be unresolved (-1)")
    tp = int(np.sum((y == 1) & (p == 1)))
    tn = int(np.sum((y == 0) & (p == 0)))
    fn = int(np.sum((y == 1) & (p != 1)))
    fp = int(np.sum((y == 0) & (p != 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    result = {
        "n": len(y),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "f2": 5 * tp / (5 * tp + fp + 4 * fn) if 5 * tp + fp + 4 * fn else 0.0,
        "accuracy": (tp + tn) / len(y),
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "unresolved_count": int(np.sum(p == -1)),
    }
    if scores is not None:
        s = np.asarray(scores, dtype=float)
        if s.shape != y.shape or not np.isfinite(s).all():
            raise ValueError("Finite aligned scores required")
        result["average_precision"] = (
            float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else None
        )
        result["roc_auc"] = float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None
    return result


def select_threshold(labels, scores) -> tuple[float, dict]:
    """Maximise validation F2 on a fixed quantile grid; ties favour fewer reviews."""
    y, s = np.asarray(labels), np.asarray(scores, dtype=float)
    if len(np.unique(y)) != 2 or not np.isfinite(s).all():
        raise ValueError("Threshold selection needs both validation classes and finite scores")
    candidates = np.unique(
        np.r_[
            np.nextafter(s.min(), -np.inf),
            np.quantile(s, np.linspace(0, 1, 101)),
            np.nextafter(s.max(), np.inf),
        ]
    )
    records = [(float(t), metrics(y, (s >= t).astype(int), s)) for t in candidates]
    return max(records, key=lambda r: (r[1]["f2"], r[1]["precision"], r[0]))


def recommendation_thresholds(labels, scores) -> dict:
    """Choose two validation cutoffs under explicit coursework costs.

    Grid-search loss: missed low-band defect=10, any manual review=1,
    normal high-band hold=3. These assumed costs need domain validation.
    """
    y, s = np.asarray(labels), np.asarray(scores)
    if len(np.unique(y)) != 2 or not np.isfinite(s).all():
        raise ValueError("Both validation classes required")
    grid = np.unique(np.r_[0.0, np.quantile(s, np.linspace(0, 1, 21)), 1.0])
    choices = []
    for low in grid:
        for high in grid:
            if low >= high:
                continue
            routine, review, hold = s < low, (s >= low) & (s < high), s >= high
            cost = (
                10 * np.sum(routine & (y == 1)) + np.sum(review) + 3 * np.sum(hold & (y == 0))
            ) / len(y)
            choices.append(
                (float(cost), float(review.mean() + hold.mean()), float(low), float(high))
            )
    cost, burden, low, high = min(choices)
    return {
        "low": low,
        "high": high,
        "assumed_costs": {
            "missed_routine_defect": 10,
            "manual_review": 1,
            "normal_priority_hold": 3,
        },
        "validation_mean_cost": cost,
        "validation_review_or_hold_fraction": burden,
        "validation_routine_missed_defects": int(np.sum((s < low) & (y == 1))),
    }


def group_bootstrap(labels, predictions, groups, repeats=1000, seed=42) -> dict:
    """Resample independent source/duplicate components, not individual patches."""
    y, p, g = np.asarray(labels), np.asarray(predictions), np.asarray(groups)
    unique = np.unique(g)
    if len(unique) < 2:
        return {"f2_95_percent_interval": None, "reason": "Fewer than two independent groups"}
    rng = np.random.default_rng(seed)
    indices = {name: np.flatnonzero(g == name) for name in unique}
    values = []
    for _ in range(repeats):
        selected = np.concatenate([indices[name] for name in rng.choice(unique, len(unique))])
        values.append(metrics(y[selected], p[selected])["f2"])
    return {
        "f2_95_percent_interval": np.quantile(values, [0.025, 0.975]).tolist(),
        "independent_groups": len(unique),
        "repeats": repeats,
        "seed": seed,
    }

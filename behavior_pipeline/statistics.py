from __future__ import annotations

"""Small tie-aware statistical helpers shared by causal evaluations."""

from collections.abc import Sequence

import numpy as np


def average_ranks(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def pearson_correlation(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
) -> float | None:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if (
        len(left_array) < 2
        or len(left_array) != len(right_array)
        or float(left_array.std()) == 0.0
        or float(right_array.std()) == 0.0
    ):
        return None
    return float(np.corrcoef(left_array, right_array)[0, 1])


def spearman_correlation(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
) -> float | None:
    if len(left) != len(right):
        raise ValueError("Spearman inputs must have equal length")
    return pearson_correlation(average_ranks(left), average_ranks(right))


def bootstrap_mean_confidence_interval(
    values: Sequence[float] | np.ndarray,
    *,
    samples: int = 2000,
    seed: int = 0,
) -> list[float] | None:
    """Return a deterministic percentile interval for a population mean."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    if array.size == 1:
        value = float(array[0])
        return [value, value]
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        means[index] = float(np.mean(rng.choice(array, size=array.size, replace=True)))
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]

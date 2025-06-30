from collections.abc import Sequence
import math
from torch import Tensor
import numpy as np


# ----------------------------------------------------------------------
# 1. Per‑layer accumulation (unchanged)
# ----------------------------------------------------------------------

def init_layer_stats():
    """Return an empty accumulator for one layer."""
    return dict(sum=0.0, sumsq=0.0, count=0,
                min=float("inf"), max=float("-inf"))


def update_layer_stats(stats: dict[str, float], tensor: Tensor):
    """
    Add a tensor’s absolute values to an existing accumulator.
    Works with NumPy arrays or PyTorch tensors.
    """
    arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    arr = np.abs(arr)

    stats["sum"]   += arr.sum()
    stats["sumsq"] += np.square(arr).sum()
    stats["count"] += arr.size
    stats["min"]    = float(min(stats["min"], arr.min()))
    stats["max"]    = float(max(stats["max"], arr.max()))
    return stats


def finalize_layer_stats(stats: dict[str, float]):
    """
    Turn raw sums into mean/std/min/max/count for a single (seed, layer) pair.
    """
    if stats["count"] == 0:
        return dict(mean=0.0, std=0.0, min=float("inf"),
                    max=float("-inf"), count=0)

    mean = stats["sum"] / stats["count"]
    var  = max(stats["sumsq"] / stats["count"] - mean**2, 0.0)
    return dict(mean=mean, std=math.sqrt(var),
                min=stats["min"], max=stats["max"],
                count=stats["count"])


# ----------------------------------------------------------------------
# 2. Correct error‑propagation across seeds
# ----------------------------------------------------------------------

def aggregate_across_seeds(means: Sequence[float], stds: Sequence[float], counts: Sequence[int] | None =None):
    """
    Combine per‑seed statistics for one layer.

    Parameters
    ----------
    means   : Sequence[float]
        Mean value from each seed.
    stds    : Sequence[float]
        Standard deviation inside each seed (same layer).
    counts  : Sequence[int] or None
        Number of elements that produced each (mean, std) pair.
        If None, all seeds are assumed to have the *same* count.

    Returns
    -------
    (global_mean, global_std)
        The layer’s mean and *pooled* standard deviation across every seed.
    """
    means = np.asarray(means, dtype=float)
    stds  = np.asarray(stds,  dtype=float)

    if counts is None:
        counts_np: np.array = np.ones_like(means, dtype=int)          # equal weight
    else:
        counts_np: np.array = np.asarray(counts_np, dtype=int)
        if counts_np.shape != means.shape:
            raise ValueError("`counts_np` must match the length of `means`/`stds`.")

    total_n   = counts_np.sum()
    if total_n == 0:
        return float("nan"), float("nan")

    # Weighted global mean μ = Σ N_i M_i / Σ N_i
    global_mean = np.average(means, weights=counts_np)

    # Sum of squares *within* seeds: Σ N_i σ_i²
    ss_within   = np.sum(counts_np * stds**2)

    # Sum of squares *between* seeds: Σ N_i (M_i - μ)²
    ss_between  = np.sum(counts_np * (means - global_mean)**2)

    # Pooled variance  σ² = (SS_within + SS_between) / Σ N_i
    pooled_var  = (ss_within + ss_between) / total_n

    return global_mean, math.sqrt(pooled_var)

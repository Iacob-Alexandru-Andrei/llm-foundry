import math
import numpy as np


def init_layer_stats():
    return {
        "sum": 0.0,
        "sumsq": 0.0,
        "count": 0,
        "min": float("inf"),
        "max": float("-inf"),
    }


def update_layer_stats(stats, tensor):
    if hasattr(tensor, "detach"):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)
    arr = np.abs(arr)
    stats["sum"] += arr.sum()
    stats["sumsq"] += np.square(arr).sum()
    stats["count"] += arr.size
    stats["min"] = float(min(stats["min"], arr.min()))
    stats["max"] = float(max(stats["max"], arr.max()))
    return stats


def finalize_layer_stats(stats):
    if stats["count"] == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": float("inf"),
            "max": float("-inf"),
            "count": 0,
        }
    mean = stats["sum"] / stats["count"]
    var = stats["sumsq"] / stats["count"] - mean**2
    var = max(var, 0.0)
    return {
        "mean": mean,
        "std": math.sqrt(var),
        "min": stats["min"],
        "max": stats["max"],
        "count": stats["count"],
    }


def aggregate_across_seeds(values, stds):
    values = np.asarray(values, dtype=float)
    stds = np.asarray(stds, dtype=float)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean_val = values.mean()
    var_between = values.var(ddof=1) if n > 1 else 0.0
    var_within = np.mean(stds ** 2)
    between_var = max(var_between - var_within, 0.0)
    total_var = (between_var + var_within) / n
    return mean_val, math.sqrt(total_var)

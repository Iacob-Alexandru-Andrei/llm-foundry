import numpy as np
from parametrisation_examples import stats_utils


def test_update_and_finalize():
    stats = stats_utils.init_layer_stats()
    arr = np.array([[1.0, -2.0], [0.0, 3.0]])
    stats_utils.update_layer_stats(stats, arr)
    result = stats_utils.finalize_layer_stats(stats)
    assert np.isclose(result["mean"], np.abs(arr).mean())
    assert np.isclose(result["std"], np.abs(arr).std())
    assert result["min"] == np.abs(arr).min()
    assert result["max"] == np.abs(arr).max()
    assert result["count"] == arr.size


def test_aggregate_across_seeds():
    vals = [1.0, 2.0, 3.0]
    stds = [0.1, 0.2, 0.3]
    mean, err = stats_utils.aggregate_across_seeds(vals, stds)
    expected_mean = np.mean(vals)
    expected_var = np.mean(np.square(stds)) / len(vals) + np.var(vals, ddof=1) / len(
        vals
    )
    assert np.isclose(mean, expected_mean)
    assert np.isclose(err, np.sqrt(expected_var))

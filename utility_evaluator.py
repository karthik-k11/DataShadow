"""Descriptive statistical comparisons for original and synthetic data.

The values returned here measure similarity only; they are not privacy
metrics, confidence intervals, or guarantees of downstream model utility.
"""

import numpy as np
import pandas as pd


def _finite_numeric(series):
    values = pd.to_numeric(series, errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    return values[np.isfinite(values)]


def _numeric_metrics(values):
    quantiles = np.quantile(values, [0.25, 0.5, 0.75])
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=0)),
        "q25": float(quantiles[0]),
        "q50": float(quantiles[1]),
        "q75": float(quantiles[2]),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _ks_statistic(first, second):
    if len(first) < 2 or len(second) < 2:
        return None
    support = np.sort(np.concatenate((first, second)))
    first_cdf = np.searchsorted(np.sort(first), support, side="right") / len(first)
    second_cdf = np.searchsorted(np.sort(second), support, side="right") / len(second)
    return float(np.max(np.abs(first_cdf - second_cdf)))


def _wasserstein_distance(first, second):
    """Empirical 1-Wasserstein distance using equally spaced quantiles."""
    quantiles = np.linspace(0, 1, max(100, min(1000, len(first) + len(second))))
    first_q = np.quantile(first, quantiles)
    second_q = np.quantile(second, quantiles)
    return float(np.mean(np.abs(first_q - second_q)))


def _category_distribution(series):
    clean = series.map(
        lambda value: ("missing",)
        if pd.isna(value)
        else ("value", type(value).__name__, repr(value))
    )
    counts = clean.value_counts(dropna=False)
    return counts / counts.sum() if counts.sum() else counts.astype(float)


def _category_key(value):
    return (type(value).__name__, repr(value))


def compare_utility(original_df, synthetic_df):
    """Compare shared numeric and categorical columns with type-appropriate metrics."""
    if not isinstance(original_df, pd.DataFrame):
        raise TypeError("original_df must be a pandas DataFrame.")
    if not isinstance(synthetic_df, pd.DataFrame):
        raise TypeError("synthetic_df must be a pandas DataFrame.")

    common_columns = [column for column in original_df.columns if column in synthetic_df.columns]
    numerical = []
    categorical = []
    missingness = []

    for column in common_columns:
        original_series = original_df[column]
        synthetic_series = synthetic_df[column]
        missingness.append({
            "column": str(column),
            "original_missing_fraction": (
                float(original_series.isna().mean()) if len(original_series) else None
            ),
            "synthetic_missing_fraction": (
                float(synthetic_series.isna().mean()) if len(synthetic_series) else None
            ),
        })
        numeric_column = (
            pd.api.types.is_numeric_dtype(original_series)
            and pd.api.types.is_numeric_dtype(synthetic_series)
        )

        if numeric_column:
            original_values = _finite_numeric(original_series)
            synthetic_values = _finite_numeric(synthetic_series)
            if not len(original_values) or not len(synthetic_values):
                numerical.append({
                    "column": str(column),
                    "available": False,
                    "reason": "No finite numeric values are available in one or both datasets.",
                })
                continue

            original_metrics = _numeric_metrics(original_values)
            synthetic_metrics = _numeric_metrics(synthetic_values)
            numerical.append({
                "column": str(column),
                "available": True,
                "original": original_metrics,
                "synthetic": synthetic_metrics,
                "mean_difference": abs(original_metrics["mean"] - synthetic_metrics["mean"]),
                "std_difference": abs(original_metrics["std"] - synthetic_metrics["std"]),
                "median_difference": abs(original_metrics["median"] - synthetic_metrics["median"]),
                "quantile_differences": {
                    key: abs(original_metrics[key] - synthetic_metrics[key])
                    for key in ("q25", "q50", "q75")
                },
                "wasserstein_distance": _wasserstein_distance(original_values, synthetic_values),
                "ks_statistic": _ks_statistic(original_values, synthetic_values),
                "notes": (
                    ["KS statistic is descriptive; small samples have low power."]
                    if min(len(original_values), len(synthetic_values)) < 20
                    else []
                ) + (
                    ["At least one distribution is constant."]
                    if np.ptp(original_values) == 0 or np.ptp(synthetic_values) == 0
                    else []
                ),
                "invalid_numeric_values": int(
                    original_series.notna().sum() - len(original_values)
                    + synthetic_series.notna().sum() - len(synthetic_values)
                ),
            })
            continue

        original_dist = _category_distribution(original_series)
        synthetic_dist = _category_distribution(synthetic_series)
        original_lookup = {_category_key(k): float(v) for k, v in original_dist.items()}
        synthetic_lookup = {_category_key(k): float(v) for k, v in synthetic_dist.items()}
        all_keys = set(original_lookup) | set(synthetic_lookup)
        total_variation = 0.5 * sum(
            abs(original_lookup.get(key, 0.0) - synthetic_lookup.get(key, 0.0))
            for key in all_keys
        )
        original_categories = set(original_lookup)
        synthetic_categories = set(synthetic_lookup)
        coverage = (
            len(original_categories & synthetic_categories) / len(original_categories)
            if original_categories
            else None
        )
        categorical.append({
            "column": str(column),
            "available": bool(all_keys),
            "original_category_count": len(original_categories),
            "synthetic_category_count": len(synthetic_categories),
            "category_coverage": coverage,
            "total_variation_distance": float(total_variation),
            "frequency_similarity": float(1.0 - total_variation),
        })

    numeric_columns = [
        column for column in common_columns
        if pd.api.types.is_numeric_dtype(original_df[column])
        and pd.api.types.is_numeric_dtype(synthetic_df[column])
    ]
    correlation = None
    if len(numeric_columns) >= 2:
        original_corr = original_df[numeric_columns].apply(pd.to_numeric, errors="coerce").corr()
        synthetic_corr = synthetic_df[numeric_columns].apply(pd.to_numeric, errors="coerce").corr()
        shared = [
            column for column in numeric_columns
            if original_corr[column].notna().all() and synthetic_corr[column].notna().all()
        ]
        if len(shared) >= 2:
            difference = np.abs(
                original_corr.loc[shared, shared].to_numpy()
                - synthetic_corr.loc[shared, shared].to_numpy()
            )
            correlation = {
                "columns": [str(column) for column in shared],
                "mean_absolute_difference": float(np.mean(difference)),
                "interpretation": "Lower is more similar; not comparable to per-column distances.",
            }

    return {
        "numerical": numerical,
        "categorical": categorical,
        "missingness": missingness,
        "correlation": correlation,
        "common_columns": [str(column) for column in common_columns],
        "definitions": {
            "mean_difference": "Absolute difference in column means; measured in the column's units.",
            "std_difference": "Absolute difference in population standard deviations; measured in column units.",
            "quantile_differences": "Absolute differences at the 25th, 50th, and 75th percentiles.",
            "wasserstein_distance": "Empirical 1-Wasserstein distance in column units; lower is more similar.",
            "ks_statistic": "Maximum absolute empirical CDF difference, ranging from 0 (identical empirical CDFs) to 1.",
            "category_coverage": "Fraction of original observed categories present in the synthetic data.",
            "total_variation_distance": "Half the L1 distance between category frequencies, from 0 to 1; lower is more similar.",
            "correlation": "Mean absolute difference between pairwise Pearson correlations; undefined for insufficient or constant columns.",
            "missingness": "Fraction of null entries in each shared column, reported separately from value-distribution metrics.",
        },
        "limitations": [
            "All metrics are descriptive and do not establish privacy, anonymity, or formal downstream model utility.",
            "Numeric distances are in source-column units and should not be compared across differently scaled columns.",
            "Small samples, missingness, and rare categories can make results unstable.",
        ],
    }

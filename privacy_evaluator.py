"""Empirical privacy-risk experiments for generated tabular data.

These comparisons are diagnostics, not attacks with calibrated success rates
and not formal privacy guarantees.
"""

import numpy as np
import pandas as pd


MAX_DISTANCE_ROWS = 500


def _canonical_value(value):
    if pd.isna(value):
        return ("missing",)
    if isinstance(value, (bool, np.bool_)):
        return ("bool", bool(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return ("number", float(value))
    return (type(value).__name__, repr(value))


def _row_key(row):
    return tuple(_canonical_value(value) for value in row)


def _column_selection(values, label):
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    if not isinstance(values, (list, tuple, set)):
        raise ValueError(f"{label} must be a column name or a list of column names.")
    return list(dict.fromkeys(values))


def _nearest_distances(original, synthetic, columns):
    original = original[columns].copy()
    synthetic = synthetic[columns].copy()
    original = original.sample(
        n=min(len(original), MAX_DISTANCE_ROWS), random_state=0
    ).reset_index(drop=True)
    synthetic = synthetic.sample(
        n=min(len(synthetic), MAX_DISTANCE_ROWS), random_state=0
    ).reset_index(drop=True)
    distance_sum = np.zeros((len(synthetic), len(original)), dtype=float)
    for column in columns:
        left = synthetic[column]
        right = original[column]
        both_missing = left.isna().to_numpy()[:, None] & right.isna().to_numpy()[None, :]
        one_missing = left.isna().to_numpy()[:, None] ^ right.isna().to_numpy()[None, :]
        if (
            pd.api.types.is_numeric_dtype(left)
            and pd.api.types.is_numeric_dtype(right)
        ):
            left_values = pd.to_numeric(left, errors="coerce").to_numpy(
                dtype=float, na_value=np.nan
            )
            right_values = pd.to_numeric(right, errors="coerce").to_numpy(
                dtype=float, na_value=np.nan
            )
            combined = np.concatenate((left_values[np.isfinite(left_values)], right_values[np.isfinite(right_values)]))
            scale = float(np.max(combined) - np.min(combined)) if len(combined) else 0.0
            if scale == 0:
                column_distance = np.zeros((len(left), len(right)), dtype=float)
            else:
                column_distance = np.abs(left_values[:, None] - right_values[None, :]) / scale
                column_distance = np.nan_to_num(column_distance, nan=0.0, posinf=1.0, neginf=1.0)
        else:
            left_values = left.map(
                lambda value: "__MISSING__" if pd.isna(value)
                else f"{type(value).__name__}:{value!r}"
            ).to_numpy(dtype=str)
            right_values = right.map(
                lambda value: "__MISSING__" if pd.isna(value)
                else f"{type(value).__name__}:{value!r}"
            ).to_numpy(dtype=str)
            column_distance = np.not_equal(left_values[:, None], right_values[None, :]).astype(float)
        column_distance[one_missing] = 1.0
        column_distance[both_missing] = 0.0
        distance_sum += column_distance
    nearest = (distance_sum / max(1, len(columns))).min(axis=1)
    return nearest, len(original), len(synthetic)


def evaluate_privacy(
    original_df,
    synthetic_df,
    quasi_identifiers=None,
    sensitive_attributes=None,
    rare_threshold=2,
):
    """Return record overlap, duplicates, proximity, rarity, and optional disclosure indicators."""
    if not isinstance(original_df, pd.DataFrame) or not isinstance(synthetic_df, pd.DataFrame):
        raise TypeError("original_df and synthetic_df must be pandas DataFrames.")
    if not isinstance(rare_threshold, int) or isinstance(rare_threshold, bool) or rare_threshold < 1:
        raise ValueError("rare_threshold must be a positive integer.")
    qis = _column_selection(quasi_identifiers, "quasi_identifiers")
    sensitive = _column_selection(sensitive_attributes, "sensitive_attributes")
    if bool(qis) != bool(sensitive):
        raise ValueError("Provide both quasi_identifiers and sensitive_attributes for disclosure analysis.")
    if set(qis) & set(sensitive):
        raise ValueError("Quasi-identifiers and sensitive attributes must be different columns.")
    if set(qis + sensitive) - set(original_df.columns):
        raise ValueError("Disclosure-analysis columns must exist in the original dataset.")
    columns = [column for column in original_df.columns if column in synthetic_df.columns]
    if not columns or original_df.empty or synthetic_df.empty:
        return {
            "experiments": [],
            "limitations": ["Comparable non-empty columns are required for privacy experiments."],
            "recommendations": ["Generate synthetic data with at least one shared non-identifier column."],
        }

    original = original_df[columns]
    synthetic = synthetic_df[columns]
    original_keys = {_row_key(row) for row in original.itertuples(index=False, name=None)}
    synthetic_keys = [_row_key(row) for row in synthetic.itertuples(index=False, name=None)]
    overlap_count = sum(key in original_keys for key in synthetic_keys)
    unique_synthetic_keys = set(synthetic_keys)
    duplicate_rows = len(synthetic_keys) - len(unique_synthetic_keys)
    experiments = [
        {
            "category": "Record overlap",
            "name": "Exact shared-column record overlap",
            "result": {
                "overlapping_synthetic_rows": int(overlap_count),
                "synthetic_row_count": int(len(synthetic)),
                "overlap_fraction": float(overlap_count / len(synthetic)),
            },
            "interpretation": "Counts synthetic rows exactly matching an original row across shared generated columns.",
            "severity": "high" if overlap_count else None,
            "limitations": "An overlap count is not a measured attack success rate; exact-match coverage depends on compared columns.",
            "recommended_mitigation": "Use independent sampling rather than source-row resampling where exact row reproduction is unacceptable.",
        },
        {
            "category": "Synthetic duplicates",
            "name": "Duplicate synthetic records",
            "result": {
                "duplicate_rows_after_first_occurrence": int(duplicate_rows),
                "synthetic_row_count": int(len(synthetic)),
                "duplicate_fraction": float(duplicate_rows / len(synthetic)),
            },
            "interpretation": "Counts repeated generated rows across shared generated columns.",
            "severity": "medium" if duplicate_rows else None,
            "limitations": "Duplicates may be expected for small or low-cardinality datasets.",
            "recommended_mitigation": "Review the generator strategy and source cardinality; do not remove duplicates blindly.",
        },
    ]

    nearest, compared_original, compared_synthetic = _nearest_distances(original, synthetic, columns)
    experiments.append({
        "category": "Similarity",
        "name": "Nearest-original mixed-type distance",
        "result": {
            "mean_nearest_distance": float(np.mean(nearest)),
            "minimum_nearest_distance": float(np.min(nearest)),
            "tenth_percentile_nearest_distance": float(np.quantile(nearest, 0.1)),
            "distance_scale": "Normalized per-column Gower-like distance from 0 (match) to 1 (maximum difference).",
            "compared_original_rows": int(compared_original),
            "compared_synthetic_rows": int(compared_synthetic),
        },
        "interpretation": "Smaller values indicate generated rows close to at least one sampled original row.",
        "severity": None,
        "limitations": [
            "At most 500 rows per dataset are compared using deterministic subsampling.",
            "Numeric dimensions use observed range scaling; categorical dimensions use equality.",
            "This proximity heuristic is not an attack or a calibrated risk probability.",
        ],
        "recommended_mitigation": "Inspect unusually small distances alongside domain context and the generation method.",
    })

    original_counts = {}
    for key in (_row_key(row) for row in original.itertuples(index=False, name=None)):
        original_counts[key] = original_counts.get(key, 0) + 1
    rare_preserved = sum(
        original_counts.get(key, 0) <= rare_threshold
        for key in synthetic_keys
        if key in original_counts
    )
    experiments.append({
        "category": "Rare records",
        "name": "Preservation of rare shared-column records",
        "result": {
            "synthetic_rows_matching_rare_originals": int(rare_preserved),
            "rare_threshold": rare_threshold,
        },
        "interpretation": "Counts exact shared-column matches to original records that occur at most the configured threshold.",
        "severity": "high" if rare_preserved else None,
        "limitations": "Rare patterns are measured only by exact shared-column match; near-rare and contextual uniqueness are not covered.",
        "recommended_mitigation": "Avoid row resampling when retaining rare original records is not acceptable.",
    })

    if qis or sensitive:
        disclosure = _attribute_disclosure(original_df, synthetic_df, qis, sensitive)
        experiments.append({
            "category": "Attribute disclosure",
            "name": "Sensitive-attribute concentration within quasi-identifier groups",
            "result": disclosure,
            "interpretation": (
                "Summarizes source groups where one sensitive value is common and how often "
                "generated rows fall into such groups; no sensitive values are included."
            ),
            "severity": None,
            "limitations": (
                "This descriptive concentration experiment does not establish actual attribute inference "
                "or attack success and depends on selected columns and the source sample."
            ),
            "recommended_mitigation": "Review quasi-identifiers and sensitive attributes with domain experts; consider suppressing or coarsening high-risk groups.",
        })

    return {
        "experiments": experiments,
        "generation_configuration": {},
        "limitations": [
            "Empirical overlap and similarity experiments do not provide differential privacy or prove anonymity.",
            "Small samples, sampling bias, column selection, and missingness can cause false positives or false negatives.",
            "No membership-inference experiment is run because this application has no trained target model or attacker protocol from which to define a meaningful attack success rate.",
        ],
        "recommendations": [
            "Treat each result as a diagnostic requiring contextual review.",
            "Do not publish data solely because these metrics appear low-risk.",
        ],
    }


def _attribute_disclosure(original_df, synthetic_df, quasi_identifiers, sensitive_attributes):
    source = original_df[quasi_identifiers + sensitive_attributes].copy()
    synthetic = synthetic_df[[c for c in quasi_identifiers if c in synthetic_df.columns]].copy()
    group_results = []
    for _, group in source.groupby(quasi_identifiers, dropna=False, observed=False):
        counts = group[sensitive_attributes].apply(
            lambda series: series.astype("object").where(series.notna(), "__MISSING__").value_counts(normalize=True).max()
        )
        concentration = float(counts.max())
        group_results.append((group.iloc[0][quasi_identifiers].tolist(), len(group), concentration))
    source_concentrations = [item[2] for item in group_results]
    matched_concentrations = []
    if len(synthetic) and all(column in synthetic.columns for column in quasi_identifiers):
        lookup = {}
        for row_values, group_size, concentration in group_results:
            key = tuple(_canonical_value(value) for value in row_values)
            lookup[key] = (group_size, concentration)
        for row in synthetic[quasi_identifiers].itertuples(index=False, name=None):
            key = tuple(_canonical_value(value) for value in row)
            if key in lookup:
                matched_concentrations.append(lookup[key][1])
    return {
        "quasi_identifier_columns": [str(column) for column in quasi_identifiers],
        "sensitive_attribute_columns": [str(column) for column in sensitive_attributes],
        "source_group_count": len(group_results),
        "source_groups_with_concentration_at_least_0_8": sum(value >= 0.8 for value in source_concentrations),
        "synthetic_rows_matching_source_groups": len(matched_concentrations),
        "mean_concentration_for_matched_groups": (
            float(np.mean(matched_concentrations)) if matched_concentrations else None
        ),
    }

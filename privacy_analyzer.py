"""Heuristic privacy-risk indicators for tabular data.

Group-size analysis is exact for the selected quasi-identifiers. It does not
measure attribute disclosure, attacker knowledge, or provide a guarantee.
"""

import re

import pandas as pd


DIRECT_IDENTIFIER_KEYWORDS = {
    "name", "email", "phone", "mobile", "address", "aadhaar", "ssn",
    "passport", "pan", "customerid", "userid", "customercode",
}


def _is_direct_identifier(column):
    normalized = re.sub(r"[^a-z0-9]", "", str(column).lower())
    return any(keyword in normalized for keyword in DIRECT_IDENTIFIER_KEYWORDS)


def analyze_privacy_risks(
    df,
    rare_threshold=2,
    uniqueness_threshold=0.9,
    quasi_identifiers=None,
    minimum_group_size=5,
):
    """Assess uniqueness, rare pairs, and group sizes for explicit QIs."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if not isinstance(rare_threshold, int) or isinstance(rare_threshold, bool) or rare_threshold < 1:
        raise ValueError("rare_threshold must be a positive integer.")
    if not isinstance(minimum_group_size, int) or isinstance(minimum_group_size, bool) or minimum_group_size < 1:
        raise ValueError("minimum_group_size must be a positive integer.")
    if (
        not isinstance(uniqueness_threshold, (int, float))
        or isinstance(uniqueness_threshold, bool)
        or not 0 < uniqueness_threshold <= 1
    ):
        raise ValueError("uniqueness_threshold must be greater than 0 and at most 1.")

    if quasi_identifiers is None:
        selected_qis = []
    elif isinstance(quasi_identifiers, str):
        selected_qis = [quasi_identifiers]
    else:
        selected_qis = list(dict.fromkeys(quasi_identifiers))
    unknown = [column for column in selected_qis if column not in df.columns]
    if unknown:
        raise ValueError("Selected quasi-identifiers must be existing columns.")

    unique_columns = []
    rare_categories = []
    rare_combinations = []
    equivalence_classes = []
    warnings = []

    for column in df.columns:
        values = df[column].dropna()
        total = len(values)
        if (
            not _is_direct_identifier(column)
            and not pd.api.types.is_numeric_dtype(df[column])
            and total
        ):
            counts = values.value_counts(dropna=False)
            rare_count = int((counts <= rare_threshold).sum())
            rare_records = int(counts[counts <= rare_threshold].sum())
            if rare_count:
                rare_categories.append({
                    "column": str(column),
                    "rare_category_count": rare_count,
                    "records_in_rare_categories": rare_records,
                    "threshold": rare_threshold,
                })
                warnings.append({
                    "type": "Rare categories",
                    "severity": "high" if rare_count and int(counts.min()) == 1 else "medium",
                    "columns": [str(column)],
                    "message": (
                        f"{rare_count} category value(s) in {column} occur at most "
                        f"{rare_threshold} time(s); rare values may make records distinguishable."
                    ),
                })
        if total >= 2:
            unique_count = int(values.nunique())
            uniqueness_ratio = unique_count / total
            if uniqueness_ratio >= uniqueness_threshold:
                unique_columns.append({
                    "column": str(column),
                    "unique_count": unique_count,
                    "total_values": total,
                    "uniqueness_ratio": round(uniqueness_ratio, 4),
                    "severity": "high" if uniqueness_ratio >= 0.98 else "medium",
                })
                warnings.append({
                    "type": "High uniqueness",
                    "severity": "high" if uniqueness_ratio >= 0.98 else "medium",
                    "columns": [str(column)],
                    "message": (
                        f"{column} has {uniqueness_ratio:.0%} distinct non-missing values "
                        "and may increase record distinguishability."
                    ),
                })

    candidate_columns = [
        column for column in df.columns
        if not _is_direct_identifier(column) and df[column].nunique(dropna=True) > 1
    ]
    for index, first in enumerate(candidate_columns):
        for second in candidate_columns[index + 1:]:
            counts = df.groupby([first, second], dropna=False, observed=False).size()
            rare_count = int((counts <= rare_threshold).sum())
            if rare_count:
                rare_combinations.append({
                    "columns": [str(first), str(second)],
                    "rare_groups": rare_count,
                    "threshold": rare_threshold,
                })
                warnings.append({
                    "type": "Rare combinations",
                    "severity": "medium",
                    "columns": [str(first), str(second)],
                    "message": (
                        f"{rare_count} value combination(s) occur at most "
                        f"{rare_threshold} time(s); they may distinguish records."
                    ),
                })

    if selected_qis and len(df):
        counts = df.groupby(selected_qis, dropna=False, observed=False).size()
        for group_size, group_count in counts.value_counts().sort_index().items():
            equivalence_classes.append({
                "group_size": int(group_size),
                "class_count": int(group_count),
                "record_count": int(group_size * group_count),
            })
        minimum_observed = int(counts.min())
        risky_groups = int((counts < minimum_group_size).sum())
        risky_records = int(counts[counts < minimum_group_size].sum())
        if risky_groups:
            warnings.append({
                "type": "Small quasi-identifier groups",
                "severity": "high" if minimum_observed == 1 else "medium",
                "columns": [str(column) for column in selected_qis],
                "message": (
                    f"{risky_records} record(s) belong to {risky_groups} group(s) "
                    f"smaller than the configured minimum size of {minimum_group_size}."
                ),
            })
        quasi_identifier_report = {
            "columns": [str(column) for column in selected_qis],
            "minimum_group_size": minimum_group_size,
            "minimum_observed_group_size": minimum_observed,
            "group_count": int(len(counts)),
            "small_group_count": risky_groups,
            "records_in_small_groups": risky_records,
            "equivalence_classes": equivalence_classes,
            "method": "Exact counts of rows sharing all selected quasi-identifier values; missing values form groups.",
            "limitation": (
                "Meeting a group-size threshold does not prevent attribute disclosure, "
                "background-knowledge attacks, or other forms of inference."
            ),
        }
    else:
        quasi_identifier_report = {
            "columns": [str(column) for column in selected_qis],
            "minimum_group_size": minimum_group_size,
            "minimum_observed_group_size": None,
            "group_count": 0,
            "small_group_count": 0,
            "records_in_small_groups": 0,
            "equivalence_classes": [],
            "method": "Select one or more quasi-identifiers to calculate exact group sizes.",
            "limitation": (
                "A group-size threshold does not address attribute disclosure or background knowledge."
            ),
        }

    return {
        "unique_columns": unique_columns,
        "rare_categories": rare_categories,
        "rare_combinations": rare_combinations,
        "quasi_identifier_analysis": quasi_identifier_report,
        "warnings": warnings,
        "thresholds": {
            "uniqueness_ratio": float(uniqueness_threshold),
            "rare_combination_count": rare_threshold,
            "minimum_group_size": minimum_group_size,
        },
        "limitations": [
            "These are heuristic indicators, not calibrated probabilities or certified privacy scores.",
            "A group-size threshold does not protect against attribute disclosure or background-knowledge attacks.",
        ],
    }

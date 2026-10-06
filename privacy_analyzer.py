import re

import pandas as pd


DIRECT_IDENTIFIER_KEYWORDS = {
    "name",
    "email",
    "phone",
    "mobile",
    "address",
    "aadhaar",
    "ssn",
    "passport",
    "pan",
}


def _is_direct_identifier(column):
    normalized = re.sub(
        r"[^a-z0-9]",
        "",
        str(column).lower(),
    )

    return any(
        keyword in normalized
        for keyword in DIRECT_IDENTIFIER_KEYWORDS
    )


def analyze_privacy_risks(df, rare_threshold=2):
    unique_columns = []
    rare_combinations = []
    warnings = []

    for column in df.columns:
        values = df[column].dropna()
        total = len(values)

        if total >= 2:
            unique_count = int(values.nunique())
            uniqueness_ratio = unique_count / total

            if uniqueness_ratio >= 0.9:
                unique_columns.append({
                    "column": str(column),
                    "unique_count": unique_count,
                    "total_values": total,
                    "uniqueness_ratio": round(
                        uniqueness_ratio, 2
                    ),
                })

                warnings.append({
                    "type": "High uniqueness",
                    "columns": [str(column)],
                    "message": (
                        "This column contains mostly unique "
                        "values and may increase identifiability."
                    ),
                })

    # Exclude likely direct identifiers from pair analysis.
    candidate_columns = [
        column
        for column in df.columns
        if not _is_direct_identifier(column)
        and df[column].nunique(dropna=True) > 1
    ]

    for i, first in enumerate(candidate_columns):
        for second in candidate_columns[i + 1:]:
            counts = (
                df.groupby(
                    [first, second],
                    dropna=False,
                )
                .size()
            )

            rare_count = int((counts <= rare_threshold).sum())

            if rare_count:
                rare_combinations.append({
                    "columns": [str(first), str(second)],
                    "rare_groups": rare_count,
                    "threshold": rare_threshold,
                })

                warnings.append({
                    "type": "Rare combinations",
                    "columns": [str(first), str(second)],
                    "message": (
                        "Some combinations occur infrequently "
                        "and may make records easier to distinguish."
                    ),
                })

    return {
        "unique_columns": unique_columns,
        "rare_combinations": rare_combinations,
        "warnings": warnings,
    }
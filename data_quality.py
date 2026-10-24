import pandas as pd


def analyze_data_quality(df):
    """Identify common data quality issues in a DataFrame."""
    total_rows = len(df)
    findings = []

    duplicate_count = int(df.duplicated().sum())

    if duplicate_count:
        findings.append({
            "type": "Duplicate Rows",
            "count": duplicate_count,
            "severity": "warning",
            "message": (
                f"{duplicate_count} duplicate row(s) detected."
            ),
        })

    if total_rows:
        for column in df.columns:
            missing_count = int(df[column].isna().sum())
            missing_percentage = missing_count / total_rows * 100

            if missing_count:
                findings.append({
                    "type": "Missing Values",
                    "column": str(column),
                    "count": missing_count,
                    "percentage": round(missing_percentage, 2),
                    "severity": (
                        "warning"
                        if missing_percentage < 50
                        else "high"
                    ),
                    "message": (
                        f"{column} contains {missing_count} missing "
                        f"value(s) ({missing_percentage:.1f}%)."
                    ),
                })

    for column in df.columns:
        if len(df) and df[column].nunique(dropna=False) == 1:
            findings.append({
                "type": "Constant Column",
                "column": str(column),
                "count": 1,
                "severity": "warning",
                "message": (
                    f"{column} contains only one distinct value."
                ),
            })

    return {
        "total_findings": len(findings),
        "duplicate_rows": duplicate_count,
        "findings": findings,
    }
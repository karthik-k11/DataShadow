import pandas as pd


def profile_dataset(df):
    """Generate basic structural statistics for a DataFrame."""

    columns = []

    for name in df.columns:
        series = df[name]

        column_info = {
            "name": str(name),
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique()),
            "numeric": pd.api.types.is_numeric_dtype(series),
            "statistics": None,
        }

        if column_info["numeric"]:
            numeric = series.dropna()

            if not numeric.empty:
                column_info["statistics"] = {
                    "mean": round(float(numeric.mean()), 2),
                    "median": round(float(numeric.median()), 2),
                    "min": round(float(numeric.min()), 2),
                    "max": round(float(numeric.max()), 2),
                }

        columns.append(column_info)

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": columns,
    }
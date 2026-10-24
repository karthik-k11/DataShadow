"""Reproducible, lightweight statistical sampling strategies."""

import re

import numpy as np
import pandas as pd


DIRECT_IDENTIFIER_KEYWORDS = {
    "name", "email", "phone", "mobile", "address", "aadhaar", "ssn",
    "passport", "pan", "customerid", "userid", "customercode",
}

GENERATION_STRATEGIES = ("independent", "joint", "stratified")
MISSING_STRATEGIES = ("preserve", "impute", "complete_cases")
NUMERICAL_STRATEGIES = ("normal", "empirical")
CATEGORICAL_STRATEGIES = ("weighted", "uniform")


def _is_direct_identifier(column):
    normalized = re.sub(r"[^a-z0-9]", "", str(column).lower())
    return any(keyword in normalized for keyword in DIRECT_IDENTIFIER_KEYWORDS)


def _validate_constraints(df, constraints):
    if constraints is None:
        constraints = {}
    if not isinstance(constraints, dict):
        raise ValueError("constraints must be a mapping of ranges and categories.")
    range_constraints = constraints.get("ranges", {})
    category_constraints = constraints.get("categories", {})
    if not isinstance(range_constraints, dict) or not isinstance(category_constraints, dict):
        raise ValueError("Range and category constraints must be mappings by column.")
    unknown = (set(range_constraints) | set(category_constraints)) - set(df.columns)
    if unknown:
        raise ValueError("Constraints must refer to existing dataset columns.")

    ranges = {}
    for column, bounds in range_constraints.items():
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError(f"Range constraint for {column} must have a minimum and maximum.")
        lower, upper = bounds
        try:
            lower = float(lower) if lower not in (None, "") else None
            upper = float(upper) if upper not in (None, "") else None
        except (TypeError, ValueError) as error:
            raise ValueError(f"Range constraint for {column} must be numeric.") from error
        if (lower is not None and not np.isfinite(lower)) or (upper is not None and not np.isfinite(upper)):
            raise ValueError(f"Range constraint for {column} must be finite.")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"Range constraint for {column} has a minimum greater than its maximum.")
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Range constraint for {column} requires a numeric column.")
        observed_values = pd.to_numeric(df[column], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        observed = pd.Series(observed_values[np.isfinite(observed_values)])
        if observed.empty or (lower is not None and upper is not None and
                              (upper < observed.min() or lower > observed.max())):
            raise ValueError(f"Range constraint for {column} does not intersect observed values.")
        ranges[column] = (lower, upper)

    categories = {}
    for column, allowed in category_constraints.items():
        if pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Category constraints cannot be applied to numeric column {column}.")
        if isinstance(allowed, str) or not isinstance(allowed, (tuple, list, set)):
            raise ValueError(f"Category constraint for {column} must contain one or more values.")
        allowed = list(dict.fromkeys(allowed))
        if not allowed:
            raise ValueError(f"Category constraint for {column} cannot be empty.")
        known_values = set(df[column].dropna().tolist())
        if not set(allowed).issubset(known_values):
            raise ValueError(f"Category constraint for {column} includes values not present in the source.")
        categories[column] = allowed
    return {"ranges": ranges, "categories": categories}


def _apply_row_constraints(df, constraints):
    mask = pd.Series(True, index=df.index)
    for column, (lower, upper) in constraints["ranges"].items():
        if lower is not None:
            mask &= pd.to_numeric(df[column], errors="coerce").ge(lower)
        if upper is not None:
            mask &= pd.to_numeric(df[column], errors="coerce").le(upper)
    for column, allowed in constraints["categories"].items():
        mask &= df[column].isin(allowed) | df[column].isna()
    return df.loc[mask]


def _sample_rows(source, row_count, rng, seed):
    if source.empty:
        raise ValueError("No source rows satisfy the selected missing-value and constraint settings.")
    return source.sample(n=row_count, replace=True, random_state=seed).reset_index(drop=True)


def _independent_sample(source, row_count, rng, numerical_strategy, categorical_strategy, missing_strategy, constraints):
    generated = {}
    for column in source.columns:
        series = source[column]
        valid = series.dropna()
        if valid.empty:
            values = np.full(row_count, np.nan, dtype=object)
        elif pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(valid, errors="coerce").dropna().to_numpy(dtype=float)
            if numerical_strategy == "empirical":
                values = rng.choice(numeric, size=row_count, replace=True)
            else:
                mean, std = float(np.mean(numeric)), float(np.std(numeric))
                values = np.full(row_count, mean) if std == 0 else rng.normal(mean, std, size=row_count)
            observed_lower, observed_upper = float(np.min(numeric)), float(np.max(numeric))
            user_lower, user_upper = constraints["ranges"].get(column, (None, None))
            lower = max(observed_lower, user_lower) if user_lower is not None else observed_lower
            upper = min(observed_upper, user_upper) if user_upper is not None else observed_upper
            if pd.api.types.is_integer_dtype(series):
                lower, upper = float(np.ceil(lower)), float(np.floor(upper))
                if lower > upper:
                    raise ValueError(
                        f"Range constraint for {column} contains no values of the column's integer type."
                    )
            if lower > upper:
                raise ValueError(f"Range constraint for {column} does not intersect generated values.")
            values = np.clip(values, lower, upper)
            if pd.api.types.is_integer_dtype(series):
                values = np.rint(values).astype("int64")
            else:
                values = values.astype(float)
        else:
            allowed = constraints["categories"].get(column)
            pool = np.array(allowed if allowed is not None else valid.to_numpy(), dtype=object)
            if categorical_strategy == "uniform":
                pool = np.array(pd.unique(pool), dtype=object)
                values = rng.choice(pool, size=row_count, replace=True)
            elif allowed is not None:
                counts = valid[valid.isin(allowed)].value_counts(normalize=True)
                weights = np.array([counts.get(value, 0.0) for value in pool], dtype=float)
                values = rng.choice(pool, size=row_count, replace=True, p=weights / weights.sum())
            elif categorical_strategy == "weighted":
                values = rng.choice(valid.to_numpy(), size=row_count, replace=True)
            else:
                values = rng.choice(pd.unique(pool), size=row_count, replace=True)
        if missing_strategy == "preserve" and series.isna().any():
            missing_rate = float(series.isna().mean())
            values = np.asarray(values, dtype=object)
            values[rng.random(row_count) < missing_rate] = np.nan
        generated[column] = values
    return pd.DataFrame(generated)


def _stratified_sample(source, row_count, rng, seed, stratify_by):
    if not stratify_by:
        raise ValueError("Select at least one column for stratified sampling.")
    if isinstance(stratify_by, str):
        stratify_by = [stratify_by]
    elif not isinstance(stratify_by, (list, tuple)):
        raise ValueError("stratify_by must be a column name or a list of column names.")
    if any(column not in source.columns for column in stratify_by):
        raise ValueError("Stratification columns must exist in the dataset.")

    groups = list(source.groupby(list(stratify_by), dropna=False, sort=False, observed=False))
    if not groups:
        raise ValueError("No source rows are available for stratified sampling.")
    counts = np.array([len(group) for _, group in groups], dtype=float)
    exact = counts / counts.sum() * row_count
    allocations = np.floor(exact).astype(int)
    for index in np.argsort(-(exact - allocations))[:row_count - int(allocations.sum())]:
        allocations[index] += 1
    sampled = [
        group.sample(
            n=int(count),
            replace=True,
            random_state=(seed + index) % 2**32,
        )
        for index, ((_, group), count) in enumerate(zip(groups, allocations))
        if count
    ]
    if not sampled:
        raise ValueError("No valid rows could be generated with the selected stratification.")
    return pd.concat(sampled, ignore_index=True).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)


def generate_synthetic_data(
    df,
    row_count=None,
    random_state=42,
    strategy="independent",
    missing_strategy="preserve",
    numerical_strategy="normal",
    categorical_strategy="weighted",
    constraints=None,
    stratify_by=None,
):
    """Generate data by independent columns, complete rows, or strata.

    Direct-identifier-named fields remain excluded for compatibility with the
    original application. This name-based heuristic is not a privacy control.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if row_count is None:
        row_count = len(df)
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 0:
        raise ValueError("row_count must be a non-negative integer.")
    if not isinstance(random_state, int) or isinstance(random_state, bool):
        raise ValueError("random_state must be an integer.")
    if not 0 <= random_state <= 2**32 - 1:
        raise ValueError("random_state must be between 0 and 4294967295.")
    if strategy not in GENERATION_STRATEGIES:
        raise ValueError(f"strategy must be one of: {', '.join(GENERATION_STRATEGIES)}.")
    if missing_strategy not in MISSING_STRATEGIES:
        raise ValueError(f"missing_strategy must be one of: {', '.join(MISSING_STRATEGIES)}.")
    if numerical_strategy not in NUMERICAL_STRATEGIES:
        raise ValueError(f"numerical_strategy must be one of: {', '.join(NUMERICAL_STRATEGIES)}.")
    if categorical_strategy not in CATEGORICAL_STRATEGIES:
        raise ValueError(f"categorical_strategy must be one of: {', '.join(CATEGORICAL_STRATEGIES)}.")
    if stratify_by and strategy != "stratified":
        raise ValueError("stratify_by can only be supplied with the stratified strategy.")
    if strategy == "stratified" and not stratify_by:
        raise ValueError("Select at least one column for stratified sampling.")

    constraints = _validate_constraints(df, constraints)
    eligible_columns = [column for column in df.columns if not _is_direct_identifier(column)]
    constrained_columns = set(constraints["ranges"]) | set(constraints["categories"])
    excluded_constraints = constrained_columns - set(eligible_columns)
    if excluded_constraints:
        raise ValueError("Constraints cannot target excluded direct-identifier columns.")
    source = df[eligible_columns].copy()
    for column in source.columns:
        if pd.api.types.is_numeric_dtype(source[column]):
            numeric = pd.to_numeric(source[column], errors="coerce").to_numpy(
                dtype=float, na_value=np.nan
            )
            invalid = ~np.isfinite(numeric)
            if invalid.any():
                source.loc[invalid, column] = np.nan
    if missing_strategy == "complete_cases":
        source = source.dropna()
    elif missing_strategy == "impute":
        for column in source.columns:
            if source[column].isna().any():
                non_missing = source[column].dropna()
                if non_missing.empty:
                    raise ValueError(
                        f"Cannot impute {column} because it contains no observed values."
                    )
                fill_value = (
                    non_missing.median()
                    if pd.api.types.is_numeric_dtype(source[column])
                    else non_missing.mode().iloc[0]
                )
                source[column] = source[column].fillna(fill_value)
    if strategy in ("joint", "stratified"):
        source = _apply_row_constraints(source, constraints)
    if not row_count:
        return pd.DataFrame(columns=eligible_columns)
    if source.empty:
        raise ValueError("No source rows satisfy the selected missing-value and constraint settings.")
    if strategy == "independent":
        for column, (lower, upper) in constraints["ranges"].items():
            observed = pd.to_numeric(source[column], errors="coerce").to_numpy(
                dtype=float, na_value=np.nan
            )
            observed = observed[np.isfinite(observed)]
            effective_lower = lower if lower is not None else (
                float(observed.min()) if len(observed) else None
            )
            effective_upper = upper if upper is not None else (
                float(observed.max()) if len(observed) else None
            )
            if not len(observed) or (
                effective_upper < observed.min()
                or effective_lower > observed.max()
            ):
                raise ValueError(
                    f"Range constraint for {column} does not intersect usable rows under the selected missing-value strategy."
                )
        for column, allowed in constraints["categories"].items():
            if not source[column].isin(allowed).any():
                raise ValueError(
                    f"Category constraint for {column} has no values after applying the selected missing-value strategy."
                )

    rng = np.random.default_rng(random_state)
    if strategy in ("joint", "stratified"):
        if strategy == "joint":
            result = _sample_rows(source, row_count, rng, random_state)
        else:
            result = _stratified_sample(source, row_count, rng, random_state, stratify_by)
    else:
        result = _independent_sample(
            source, row_count, rng, "normal" if numerical_strategy == "normal" else "empirical",
            categorical_strategy, missing_strategy, constraints,
        )
    return result.reindex(columns=eligible_columns)


def describe_generation(strategy):
    descriptions = {
        "independent": (
            "Samples each column independently; per-column patterns are approximated, "
            "but relationships between columns may be lost."
        ),
        "joint": (
            "Resamples complete source rows and therefore preserves observed relationships "
            "but can reproduce source records exactly. Per-column numerical and categorical "
            "sampling choices are not applied by this strategy."
        ),
        "stratified": (
            "Resamples complete rows within selected groups to approximate their source proportions; "
            "it can preserve rare groups and source records. Per-column numerical and categorical "
            "sampling choices are not applied by this strategy."
        ),
    }
    if strategy not in descriptions:
        raise ValueError("Unknown generation strategy.")
    return descriptions[strategy]

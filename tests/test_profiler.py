import pandas as pd

from profiler import profile_dataset


def test_basic_profile():
    df = pd.DataFrame({
        "name": ["Rahul", "Priya", "Arun"],
        "age": [28, 31, 28],
        "city": ["Bengaluru", "Mysuru", "Bengaluru"],
    })

    result = profile_dataset(df)

    assert result["row_count"] == 3
    assert result["column_count"] == 3
    assert result["missing_total"] == 0
    assert result["duplicate_rows"] == 0


def test_missing_values():
    df = pd.DataFrame({
        "name": ["Rahul", None],
        "age": [28, 31],
    })

    result = profile_dataset(df)

    assert result["missing_total"] == 1
    assert result["columns"][0]["missing"] == 1


def test_duplicate_rows():
    df = pd.DataFrame({
        "name": ["Rahul", "Rahul", "Priya"],
        "age": [28, 28, 31],
    })

    result = profile_dataset(df)

    assert result["duplicate_rows"] == 1


def test_numeric_statistics():
    df = pd.DataFrame({"age": [28, 31, 28]})

    result = profile_dataset(df)
    stats = result["columns"][0]["statistics"]

    assert stats["mean"] == 29.0
    assert stats["median"] == 28.0
    assert stats["min"] == 28.0
    assert stats["max"] == 31.0


def test_empty_dataframe():
    df = pd.DataFrame({"age": pd.Series(dtype="float64")})

    result = profile_dataset(df)

    assert result["row_count"] == 0
    assert result["column_count"] == 1
    assert result["columns"][0]["statistics"] is None
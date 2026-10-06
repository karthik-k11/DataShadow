import pandas as pd

from privacy_analyzer import analyze_privacy_risks


def test_detects_high_uniqueness():
    df = pd.DataFrame({
        "customer_code": ["A1", "A2", "A3", "A4"],
        "age": [25, 25, 30, 30],
    })

    result = analyze_privacy_risks(df)
    columns = [
        item["column"]
        for item in result["unique_columns"]
    ]

    assert "customer_code" in columns
    assert "age" not in columns


def test_detects_rare_combinations():
    df = pd.DataFrame({
        "age": [25, 25, 30, 40],
        "city": ["Bengaluru", "Bengaluru", "Mysuru", "Hubballi"],
    })

    result = analyze_privacy_risks(df)

    combinations = [
        item["columns"]
        for item in result["rare_combinations"]
    ]

    assert ["age", "city"] in combinations


def test_excludes_direct_identifiers_from_combinations():
    df = pd.DataFrame({
        "name": ["Rahul", "Priya", "Arun"],
        "age": [25, 30, 35],
        "city": ["Bengaluru", "Mysuru", "Hubballi"],
    })

    result = analyze_privacy_risks(df)

    for item in result["rare_combinations"]:
        assert "name" not in item["columns"]


def test_handles_empty_dataframe():
    df = pd.DataFrame({
        "age": pd.Series(dtype="int64"),
        "city": pd.Series(dtype="object"),
    })

    result = analyze_privacy_risks(df)

    assert result["unique_columns"] == []
    assert result["rare_combinations"] == []
    assert result["warnings"] == []
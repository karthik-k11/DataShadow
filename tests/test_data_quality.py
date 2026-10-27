import pandas as pd

from data_quality import analyze_data_quality


def test_detects_duplicate_rows():
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Alice"],
        "age": [25, 30, 25],
    })

    result = analyze_data_quality(df)

    assert result["duplicate_rows"] == 1


def test_detects_missing_values():
    df = pd.DataFrame({
        "name": ["Alice", None, "Charlie"],
    })

    result = analyze_data_quality(df)

    assert any(
        item["type"] == "Missing Values"
        and item["column"] == "name"
        and item["count"] == 1
        for item in result["findings"]
    )


def test_detects_constant_columns():
    df = pd.DataFrame({
        "country": ["India", "India", "India"],
        "name": ["Alice", "Bob", "Charlie"],
    })

    result = analyze_data_quality(df)

    assert any(
        item["type"] == "Constant Column"
        and item["column"] == "country"
        for item in result["findings"]
    )


def test_clean_dataset_has_no_findings():
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
    })

    result = analyze_data_quality(df)

    assert result["total_findings"] == 0


def test_empty_dataframe_is_handled():
    df = pd.DataFrame(columns=["name", "age"])

    result = analyze_data_quality(df)

    assert result["duplicate_rows"] == 0
    assert result["total_findings"] == 0
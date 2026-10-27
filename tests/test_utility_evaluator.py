import pandas as pd
import pytest

from utility_evaluator import compare_utility


def test_compares_numeric_columns():
    original = pd.DataFrame({
        "age": [20, 30, 40, 50],
    })

    synthetic = pd.DataFrame({
        "age": [22, 31, 39, 48],
    })

    result = compare_utility(original, synthetic)

    assert len(result["numerical"]) == 1
    assert result["numerical"][0]["column"] == "age"
    assert "mean_difference" in result["numerical"][0]
    assert "median_difference" in result["numerical"][0]


def test_compares_categorical_columns():
    original = pd.DataFrame({
        "city": [
            "Bengaluru",
            "Mysuru",
            "Bengaluru",
            "Hubballi",
        ],
    })

    synthetic = pd.DataFrame({
        "city": [
            "Bengaluru",
            "Mysuru",
            "Bengaluru",
            "Bengaluru",
        ],
    })

    result = compare_utility(original, synthetic)

    assert len(result["categorical"]) == 1
    assert result["categorical"][0]["column"] == "city"
    assert result["categorical"][0]["category_coverage"] == pytest.approx(2 / 3)


def test_only_common_columns_are_compared():
    original = pd.DataFrame({
        "age": [20, 30],
        "city": ["Bengaluru", "Mysuru"],
    })

    synthetic = pd.DataFrame({
        "age": [21, 31],
        "income": [50000, 60000],
    })

    result = compare_utility(original, synthetic)

    assert result["common_columns"] == ["age"]
    assert len(result["numerical"]) == 1


def test_rejects_non_dataframe_input():
    with pytest.raises(TypeError):
        compare_utility([], pd.DataFrame({"age": [20]}))

    with pytest.raises(TypeError):
        compare_utility(pd.DataFrame({"age": [20]}), [])


def test_handles_missing_values():
    original = pd.DataFrame({
        "age": [20, None, 40],
    })

    synthetic = pd.DataFrame({
        "age": [22, 32, None],
    })

    result = compare_utility(original, synthetic)

    assert len(result["numerical"]) == 1
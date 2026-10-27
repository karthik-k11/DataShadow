import numpy as np
import pandas as pd
import pytest

from utility_evaluator import compare_utility


def test_numeric_report_has_distribution_and_quantile_metrics():
    source = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 2.0, 4.0, 6.0]})
    same = source.copy()

    report = compare_utility(source, same)

    x = report["numerical"][0]
    assert x["mean_difference"] == 0
    assert x["std_difference"] == 0
    assert x["quantile_differences"]["q25"] == 0
    assert x["wasserstein_distance"] == 0
    assert x["ks_statistic"] == 0
    assert report["correlation"]["mean_absolute_difference"] == 0


def test_numeric_distances_capture_distribution_shift_and_constants():
    report = compare_utility(
        pd.DataFrame({"constant": [4, 4, 4], "x": [0, 1, 2]}),
        pd.DataFrame({"constant": [5, 5, 5], "x": [2, 3, 4]}),
    )

    assert report["numerical"][0]["wasserstein_distance"] == pytest.approx(1)
    assert any("constant" in note for note in report["numerical"][0]["notes"])
    assert report["numerical"][1]["ks_statistic"] > 0
    assert report["correlation"] is None


def test_categorical_tvd_and_missingness_are_reported():
    report = compare_utility(
        pd.DataFrame({"category": ["a", "a", "b", None]}),
        pd.DataFrame({"category": ["a", "b", "b", None]}),
    )

    categorical = report["categorical"][0]
    assert categorical["category_coverage"] == 1
    assert categorical["total_variation_distance"] == pytest.approx(0.25)
    assert categorical["frequency_similarity"] == pytest.approx(0.75)
    assert report["missingness"][0]["original_missing_fraction"] == pytest.approx(0.25)
    assert report["missingness"][0]["synthetic_missing_fraction"] == pytest.approx(0.25)


def test_small_and_all_invalid_numeric_columns_are_not_fabricated():
    report = compare_utility(
        pd.DataFrame({"x": [np.nan, np.inf]}),
        pd.DataFrame({"x": [1.0, 2.0]}),
    )

    assert report["numerical"][0]["available"] is False
    assert report["numerical"][0]["reason"]


def test_utility_report_documents_separate_metric_scales():
    report = compare_utility(pd.DataFrame({"x": [1, 2]}), pd.DataFrame({"x": [1, 2]}))

    assert "limitations" in report
    assert "wasserstein_distance" in report["definitions"]


def test_missing_category_marker_cannot_collide_with_a_real_value():
    report = compare_utility(
        pd.DataFrame({"category": ["__DATASHADOW_MISSING__", None]}),
        pd.DataFrame({"category": ["__DATASHADOW_MISSING__", "__DATASHADOW_MISSING__"]}),
    )

    assert report["categorical"][0]["category_coverage"] == pytest.approx(0.5)
    assert report["categorical"][0]["total_variation_distance"] == pytest.approx(0.5)


def test_nullable_numeric_dtypes_are_supported():
    original = pd.DataFrame({"x": pd.Series([1, None, 3], dtype="Int64")})
    synthetic = pd.DataFrame({"x": pd.Series([2, 3, None], dtype="Int64")})

    report = compare_utility(original, synthetic)

    assert report["numerical"][0]["available"]
    assert report["missingness"][0]["original_missing_fraction"] == pytest.approx(1 / 3)

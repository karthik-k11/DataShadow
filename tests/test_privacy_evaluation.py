import pandas as pd
import pytest

from privacy_analyzer import analyze_privacy_risks
from privacy_evaluator import evaluate_privacy


def test_quasi_identifier_analysis_reports_exact_group_sizes():
    source = pd.DataFrame({
        "age_band": ["20s", "20s", "30s", "30s", "30s"],
        "region": ["n", "n", "s", "s", "s"],
    })

    report = analyze_privacy_risks(
        source,
        quasi_identifiers=["age_band", "region"],
        minimum_group_size=2,
    )

    analysis = report["quasi_identifier_analysis"]
    assert analysis["group_count"] == 2
    assert analysis["minimum_observed_group_size"] == 2
    assert analysis["small_group_count"] == 0
    assert sum(item["record_count"] for item in analysis["equivalence_classes"]) == 5


def test_quasi_identifier_singletons_are_flagged():
    source = pd.DataFrame({"qid": ["a", "b", "c"]})
    report = analyze_privacy_risks(
        source, quasi_identifiers=["qid"], minimum_group_size=2
    )

    assert report["quasi_identifier_analysis"]["records_in_small_groups"] == 3
    assert report["warnings"][-1]["severity"] == "high"


def test_detects_rare_categories_without_exposing_values():
    source = pd.DataFrame({"group": ["common", "common", "rare-secret"]})

    report = analyze_privacy_risks(source, rare_threshold=1)

    assert report["rare_categories"] == [{
        "column": "group",
        "rare_category_count": 1,
        "records_in_rare_categories": 1,
        "threshold": 1,
    }]
    assert "rare-secret" not in str(report)


@pytest.mark.parametrize(
    "options",
    [
        {"rare_threshold": 0},
        {"uniqueness_threshold": 0},
        {"minimum_group_size": 0},
        {"quasi_identifiers": ["missing_column"]},
    ],
)
def test_privacy_analysis_rejects_invalid_settings(options):
    with pytest.raises(ValueError):
        analyze_privacy_risks(pd.DataFrame({"x": [1, 2]}), **options)


def test_privacy_experiments_measure_overlap_duplicates_and_distance():
    original = pd.DataFrame({"x": [1, 2], "group": ["a", "b"]})
    synthetic = pd.DataFrame({"x": [1, 1, 3], "group": ["a", "a", "c"]})

    report = evaluate_privacy(original, synthetic)
    experiments = {item["category"]: item for item in report["experiments"]}

    assert experiments["Record overlap"]["result"]["overlapping_synthetic_rows"] == 2
    assert experiments["Synthetic duplicates"]["result"]["duplicate_rows_after_first_occurrence"] == 1
    assert experiments["Similarity"]["result"]["minimum_nearest_distance"] == 0


def test_attribute_disclosure_report_does_not_include_sensitive_values():
    original = pd.DataFrame({
        "region": ["north", "north", "south", "south"],
        "condition": ["secret-a", "secret-a", "secret-b", "secret-c"],
    })
    synthetic = original[["region"]].copy()

    report = evaluate_privacy(
        original,
        synthetic,
        quasi_identifiers=["region"],
        sensitive_attributes=["condition"],
    )

    result = report["experiments"][-1]["result"]
    assert result["source_group_count"] == 2
    assert result["source_groups_with_concentration_at_least_0_8"] == 1
    assert "secret-a" not in str(report)
    assert "secret-b" not in str(report)


def test_privacy_evaluator_handles_empty_data_and_invalid_disclosure():
    result = evaluate_privacy(
        pd.DataFrame({"x": []}), pd.DataFrame({"x": []})
    )
    assert result["experiments"] == []
    report = evaluate_privacy(
        pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [2]})
    )
    assert any("No membership-inference experiment" in note for note in report["limitations"])

    with pytest.raises(ValueError, match="both quasi_identifiers"):
        evaluate_privacy(
            pd.DataFrame({"x": [1], "s": ["a"]}),
            pd.DataFrame({"x": [1]}),
            sensitive_attributes=["s"],
        )

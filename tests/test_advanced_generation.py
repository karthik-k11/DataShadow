import pandas as pd
import pytest

from synthetic_generator import generate_synthetic_data


@pytest.fixture
def mixed_data():
    return pd.DataFrame({
        "age": [20, 30, 40, 50],
        "group": ["a", "a", "b", "b"],
        "city": ["x", "y", "x", "y"],
    })


@pytest.mark.parametrize("strategy", ["independent", "joint", "stratified"])
def test_all_strategies_are_reproducible(mixed_data, strategy):
    options = {"stratify_by": ["group"]} if strategy == "stratified" else {}
    first = generate_synthetic_data(
        mixed_data, 25, 17, strategy=strategy, **options
    )
    second = generate_synthetic_data(
        mixed_data, 25, 17, strategy=strategy, **options
    )

    pd.testing.assert_frame_equal(first, second)


def test_joint_strategy_preserves_complete_source_rows(mixed_data):
    generated = generate_synthetic_data(
        mixed_data, 50, 4, strategy="joint"
    )
    source_rows = set(map(tuple, mixed_data.to_numpy().tolist()))

    assert set(map(tuple, generated.to_numpy().tolist())).issubset(source_rows)


def test_stratified_strategy_preserves_group_proportions(mixed_data):
    generated = generate_synthetic_data(
        mixed_data, 100, strategy="stratified", stratify_by=["group"]
    )

    assert generated["group"].value_counts(normalize=True).to_dict() == {
        "a": 0.5,
        "b": 0.5,
    }


def test_stratified_strategy_accepts_maximum_supported_seed(mixed_data):
    generated = generate_synthetic_data(
        mixed_data,
        5,
        random_state=2**32 - 1,
        strategy="stratified",
        stratify_by=["group"],
    )

    assert len(generated) == 5


def test_numeric_and_categorical_constraints_are_respected(mixed_data):
    generated = generate_synthetic_data(
        mixed_data,
        50,
        constraints={
            "ranges": {"age": [25, 35]},
            "categories": {"city": ["x"]},
        },
    )

    assert generated["age"].between(25, 35).all()
    assert set(generated["city"]) == {"x"}


def test_integer_ranges_reject_empty_integer_intersections(mixed_data):
    with pytest.raises(ValueError, match="integer type"):
        generate_synthetic_data(
            mixed_data,
            5,
            constraints={"ranges": {"age": [30.1, 30.9]}},
        )


def test_weighted_categorical_constraints_preserve_relative_frequencies():
    source = pd.DataFrame({"category": ["frequent"] * 9 + ["rare"]})

    generated = generate_synthetic_data(
        source,
        1000,
        random_state=4,
        constraints={"categories": {"category": ["frequent", "rare"]}},
    )

    assert (generated["category"] == "frequent").mean() > 0.8


@pytest.mark.parametrize(
    "options",
    [
        {"strategy": "invalid"},
        {"missing_strategy": "invalid"},
        {"numerical_strategy": "invalid"},
        {"categorical_strategy": "invalid"},
        {"row_count": -1},
        {"random_state": -1},
        {"strategy": "stratified"},
        {"stratify_by": ["group"]},
        {"constraints": {"ranges": {"age": [40, 20]}}},
        {"constraints": {"ranges": {"age": [100, None]}}},
        {"constraints": {"categories": {"city": ["not present"]}}},
    ],
)
def test_invalid_generation_settings_are_rejected(mixed_data, options):
    with pytest.raises(ValueError):
        generate_synthetic_data(mixed_data, **options)


def test_missing_value_modes(mixed_data):
    source = mixed_data.copy()
    source.loc[0, "age"] = None

    preserved = generate_synthetic_data(source, 100, random_state=2)
    imputed = generate_synthetic_data(
        source, 100, random_state=2, missing_strategy="impute"
    )
    complete = generate_synthetic_data(
        source, 100, random_state=2, strategy="joint",
        missing_strategy="complete_cases"
    )

    assert preserved["age"].isna().any()
    assert not imputed["age"].isna().any()
    assert not complete["age"].isna().any()


def test_fully_missing_column_cannot_be_imputed():
    source = pd.DataFrame({"x": [None, None]})

    with pytest.raises(ValueError, match="no observed values"):
        generate_synthetic_data(source, 2, missing_strategy="impute")


def test_category_constraint_conflicting_with_complete_case_filter_is_rejected():
    source = pd.DataFrame({
        "category": ["rare", "common", "common"],
        "value": [None, 1.0, 2.0],
    })

    with pytest.raises(ValueError, match="no values after applying"):
        generate_synthetic_data(
            source,
            4,
            missing_strategy="complete_cases",
            constraints={"categories": {"category": ["rare"]}},
        )

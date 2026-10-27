import pandas as pd

from synthetic_generator import generate_synthetic_data


def test_generates_requested_row_count():
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "city": ["Bengaluru", "Mysuru", "Hubballi"],
    })

    result = generate_synthetic_data(df, row_count=10)

    assert len(result) == 10
    assert list(result.columns) == ["age", "city"]


def test_excludes_direct_identifier_columns():
    df = pd.DataFrame({
        "name": ["Rahul", "Priya", "Arun"],
        "email": ["a@example.com", "b@example.com", "c@example.com"],
        "age": [20, 30, 40],
    })

    result = generate_synthetic_data(df)

    assert "name" not in result.columns
    assert "email" not in result.columns
    assert "age" in result.columns


def test_generation_is_reproducible():
    df = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "city": ["Bengaluru", "Mysuru", "Hubballi", "Bengaluru"],
    })

    first = generate_synthetic_data(df, random_state=42)
    second = generate_synthetic_data(df, random_state=42)

    pd.testing.assert_frame_equal(first, second)


def test_handles_empty_dataframe():
    df = pd.DataFrame({
        "age": pd.Series(dtype="float64"),
    })

    result = generate_synthetic_data(df)

    assert result.empty
    assert list(result.columns) == ["age"]


def test_rejects_negative_row_count():
    df = pd.DataFrame({"age": [20, 30, 40]})

    try:
        generate_synthetic_data(df, row_count=-1)
        assert False, "Expected ValueError"
    except ValueError:
        pass
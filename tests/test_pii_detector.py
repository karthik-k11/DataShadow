import pandas as pd

from pii_detector import detect_pii


def test_detects_sensitive_column_names():
    df = pd.DataFrame({
        "name": ["Rahul"],
        "email": ["rahul@example.com"],
        "phone": ["9876543210"],
        "city": ["Bengaluru"],
    })

    findings = detect_pii(df)

    detected = {item["column"]: item["type"] for item in findings}

    assert detected["name"] == "Potential Name"
    assert detected["email"] == "Potential Email"
    assert detected["phone"] == "Potential Phone Number"
    assert "city" not in detected


def test_detects_email_by_value_pattern():
    df = pd.DataFrame({
        "contact": [
            "rahul@example.com",
            "priya@example.com",
            "arun@example.com",
        ]
    })

    findings = detect_pii(df)

    assert findings[0]["type"] == "Potential Email"


def test_ignores_normal_columns():
    df = pd.DataFrame({
        "city": ["Bengaluru", "Mysuru"],
        "department": ["Sales", "Engineering"],
    })

    assert detect_pii(df) == []


def test_handles_empty_dataframe():
    df = pd.DataFrame({"email": pd.Series(dtype="object")})

    assert detect_pii(df) == []
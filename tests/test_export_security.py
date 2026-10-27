
from io import BytesIO
import time

import pandas as pd

import app as app_module


def test_csv_export_escapes_formula_prefixes():
    df = pd.DataFrame({
        "value": [
            "=SUM(A1:A2)",
            "+cmd",
            "-formula",
            "@reference",
            "\tformula",
            "\rformula",
            "normal text",
        ]
    })

    csv_text = app_module.safe_csv_bytes(df).decode("utf-8")
    exported = pd.read_csv(BytesIO(csv_text.encode("utf-8")))

    expected = [
        "'=SUM(A1:A2)",
        "'+cmd",
        "'-formula",
        "'@reference",
        "'\tformula",
        "'\rformula",
        "normal text",
    ]

    assert exported["value"].tolist() == expected


def test_csv_export_does_not_modify_numeric_values():
    df = pd.DataFrame({"amount": [-100, 0, 250]})

    exported = pd.read_csv(
        BytesIO(app_module.safe_csv_bytes(df))
    )

    assert exported["amount"].tolist() == [-100, 0, 250]


def test_synthetic_download_returns_csv(monkeypatch):
    token = "test-download-token"
    monkeypatch.setitem(
        app_module.SYNTHETIC_EXPORTS,
        token,
        b"name,amount\nAlice,100\n",
    )

    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    response = client.get(f"/download-synthetic/{token}")

    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    assert b"Alice,100" in response.data
    assert "attachment" in response.headers["Content-Disposition"]


def test_synthetic_download_token_can_only_be_used_once(monkeypatch):
    token = "single-use-token"
    monkeypatch.setitem(
        app_module.SYNTHETIC_EXPORTS,
        token,
        b"name\nAlice\n",
    )

    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    first_response = client.get(f"/download-synthetic/{token}")
    second_response = client.get(f"/download-synthetic/{token}")

    assert first_response.status_code == 200
    assert second_response.status_code == 404


def test_unknown_synthetic_download_token_returns_404():
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    response = client.get("/download-synthetic/unknown-token")

    assert response.status_code == 404


def test_expired_synthetic_download_token_returns_404(monkeypatch):
    token = "expired-download-token"
    monkeypatch.setitem(
        app_module.SYNTHETIC_EXPORTS, token, b"value\nsample\n"
    )
    monkeypatch.setitem(
        app_module.SYNTHETIC_EXPORT_EXPIRY, token, time.time() - 1
    )
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    assert client.get(f"/download-synthetic/{token}").status_code == 404


def test_expired_privacy_report_token_returns_404(monkeypatch):
    token = "expired-report-token"
    monkeypatch.setitem(app_module.REPORT_EXPORTS, token, b"metric,result\n")
    monkeypatch.setitem(
        app_module.REPORT_EXPORT_EXPIRY, token, time.time() - 1
    )
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    assert client.get(f"/download-privacy-report/{token}").status_code == 404


def test_non_csv_upload_is_rejected():
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    response = client.post(
        "/",
        data={"file": (BytesIO(b"name\nAlice\n"), "data.txt")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"Only CSV files are allowed" in response.data
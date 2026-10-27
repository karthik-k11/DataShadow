from io import BytesIO
import re

import app as app_module


def _client(monkeypatch):
    monkeypatch.setattr(app_module, "get_history", lambda limit=20: [])
    monkeypatch.setattr(app_module, "save_analysis", lambda **kwargs: 1)
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_upload_generate_compare_and_report_export_workflow(monkeypatch):
    client = _client(monkeypatch)
    upload = client.post(
        "/",
        data={
            "file": (
                BytesIO(b"age,city,group\n20,north,a\n30,south,a\n40,north,b\n50,south,b\n"),
                "sample.csv",
            ),
            "action": "analyze",
        },
        content_type="multipart/form-data",
    )
    assert upload.status_code == 200

    generated = client.post(
        "/",
        data={
            "action": "generate",
            "row_count": "12",
            "random_seed": "19",
            "strategy": "independent",
            "missing_strategy": "preserve",
            "numerical_strategy": "empirical",
            "categorical_strategy": "weighted",
            "quasi_identifiers": ["group"],
            "sensitive_attributes": ["city"],
            "minimum_group_size": "2",
            "rare_threshold": "2",
            "uniqueness_threshold": "0.9",
        },
    )
    assert generated.status_code == 200
    assert b"Empirical privacy evaluation" in generated.data
    assert b"Consolidated privacy" in generated.data
    assert b"Download evaluation CSV" in generated.data

    token_match = re.search(rb"/download-privacy-report/([A-Za-z0-9_-]+)", generated.data)
    assert token_match
    response = client.get(
        "/download-privacy-report/" + token_match.group(1).decode("ascii")
    )
    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    assert b"Exact shared-column record overlap" in response.data
    assert "attachment" in response.headers["Content-Disposition"]


def test_invalid_generation_setting_is_shown_without_losing_active_dataset(monkeypatch):
    client = _client(monkeypatch)
    client.post(
        "/",
        data={
            "file": (BytesIO(b"age,city\n20,north\n30,south\n"), "sample.csv"),
            "action": "analyze",
        },
        content_type="multipart/form-data",
    )

    response = client.post(
        "/",
        data={
            "action": "generate",
            "row_count": "4",
            "random_seed": "-1",
        },
    )

    assert response.status_code == 200
    assert b"Random seed must be between" in response.data
    assert b"sample.csv" in response.data


def test_json_category_constraint_is_applied_and_reflected(monkeypatch):
    client = _client(monkeypatch)
    client.post(
        "/",
        data={
            "file": (
                BytesIO(b"age,city\n20,\"New, York\"\n30,north\n40,north\n"),
                "sample.csv",
            ),
            "action": "analyze",
        },
        content_type="multipart/form-data",
    )

    response = client.post(
        "/",
        data={
            "action": "generate",
            "row_count": "8",
            "random_seed": "7",
            "strategy": "independent",
            "category_values_1": '["New, York"]',
        },
    )

    assert response.status_code == 200
    assert b"New, York" in response.data
    assert b"category_values_1" in response.data


def test_privacy_configuration_is_validated_and_persisted_in_session(monkeypatch):
    client = _client(monkeypatch)
    client.post(
        "/",
        data={
            "file": (BytesIO(b"age,city\n20,north\n30,south\n"), "sample.csv"),
            "action": "analyze",
        },
        content_type="multipart/form-data",
    )
    response = client.post(
        "/",
        data={
            "action": "privacy",
            "quasi_identifiers": ["city"],
            "minimum_group_size": "2",
            "rare_threshold": "2",
            "uniqueness_threshold": "0.8",
        },
    )

    assert response.status_code == 200
    assert b"Selected columns: city" in response.data
    assert b"0.8" in response.data

    invalid = client.post(
        "/",
        data={
            "action": "privacy",
            "minimum_group_size": "0",
        },
    )
    assert b"Minimum group size must be at least 1" in invalid.data


def test_invalid_privacy_report_export_token_is_rejected(monkeypatch):
    client = _client(monkeypatch)
    assert client.get("/download-privacy-report/not-a-valid-token").status_code == 404

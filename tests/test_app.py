from io import BytesIO

import app as app_module


def test_dashboard_loads_successfully(monkeypatch):
    monkeypatch.setattr(app_module, "get_history", lambda limit=20: [])

    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b"DataShadow" in response.data
    assert b"Dashboard Summary" in response.data
    assert b"Analysis History" in response.data


def test_dashboard_displays_empty_history(monkeypatch):
    monkeypatch.setattr(app_module, "get_history", lambda limit=20: [])

    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b"No analysis history yet" in response.data


def test_clear_history_redirects_to_dashboard(monkeypatch):
    monkeypatch.setattr(app_module, "clear_history", lambda: 3)
    monkeypatch.setattr(app_module, "get_history", lambda limit=20: [])

    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    response = client.post("/clear-history")

    assert response.status_code == 302
    assert "/?history_cleared=3" in response.headers["Location"]


def test_clear_history_displays_success_message(monkeypatch):
    monkeypatch.setattr(app_module, "clear_history", lambda: 3)
    monkeypatch.setattr(app_module, "get_history", lambda limit=20: [])

    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    response = client.post("/clear-history", follow_redirects=True)

    assert response.status_code == 200
    assert b"Analysis history cleared" in response.data
    assert b"3 record(s) removed" in response.data


def test_clear_empty_history_displays_zero_records(monkeypatch):
    monkeypatch.setattr(app_module, "clear_history", lambda: 0)
    monkeypatch.setattr(app_module, "get_history", lambda limit=20: [])

    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    response = client.post("/clear-history", follow_redirects=True)

    assert response.status_code == 200
    assert b"Analysis history cleared" in response.data
    assert b"0 record(s) removed" in response.data


def test_empty_csv_is_rejected():
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    response = client.post(
        "/",
        data={"file": (BytesIO(b""), "empty.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"Unable to read this file as a valid CSV" in response.data


def test_malformed_csv_is_rejected():
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    response = client.post(
        "/",
        data={
            "file": (
                BytesIO(b'a,b\n"unterminated,2\n'),
                "malformed.csv",
            )
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"Unable to read this file as a valid CSV" in response.data


def test_csv_exceeding_row_limit_is_rejected():
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    csv_content = (
        "value\n" + "1\n" * (app_module.MAX_ROWS + 1)
    ).encode("utf-8")

    response = client.post(
        "/",
        data={"file": (BytesIO(csv_content), "too_many_rows.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"cannot exceed 10000 rows" in response.data


def test_csv_exceeding_column_limit_is_rejected():
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    headers = ",".join(
        f"column_{i}" for i in range(app_module.MAX_COLUMNS + 1)
    )
    values = ",".join("value" for _ in range(app_module.MAX_COLUMNS + 1))
    csv_content = f"{headers}\n{values}\n".encode("utf-8")

    response = client.post(
        "/",
        data={"file": (BytesIO(csv_content), "too_many_columns.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"cannot exceed 100 columns" in response.data


def test_upload_exceeding_file_size_limit_is_rejected():
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    oversized_content = b"a" * (
        app_module.app.config["MAX_CONTENT_LENGTH"] + 1
    )

    response = client.post(
        "/",
        data={"file": (BytesIO(oversized_content), "oversized.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 413
    assert b"5 MB upload limit" in response.data

def test_duplicate_column_names_are_handled():
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    response = client.post(
        "/",
        data={
            "file": (
                BytesIO(b"name,name\nAlice,25\nBob,30\n"),
                "duplicate_columns.csv",
            )
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"unexpected error" not in response.data.lower()


def test_csv_with_missing_values_is_processed():
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    response = client.post(
        "/",
        data={
            "file": (
                BytesIO(b"name,age,city\nAlice,25,\nBob,,Bengaluru\n"),
                "missing_values.csv",
            )
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"unexpected error" not in response.data.lower()


def test_unsupported_text_encoding_is_rejected():
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    response = client.post(
        "/",
        data={
            "file": (
                BytesIO(b"name\n\xff\xfe\xfa"),
                "invalid_encoding.csv",
            )
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"Unable to read this file as a valid CSV" in response.data


def test_csv_with_only_headers_is_handled():
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    response = client.post(
        "/",
        data={
            "file": (BytesIO(b"name,age,city\n"), "headers_only.csv")
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"unexpected error" not in response.data.lower()
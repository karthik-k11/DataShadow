
import sqlite3

import pytest

from history_db import (
    clear_history,
    get_history,
    init_db,
    save_analysis,
)


def test_init_db_creates_history_table(tmp_path):
    db_path = tmp_path / "history.db"

    init_db(db_path)

    with sqlite3.connect(db_path) as connection:
        tables = connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name = 'analysis_history'
            """
        ).fetchall()

    assert tables == [("analysis_history",)]


def test_save_and_retrieve_analysis(tmp_path):
    db_path = tmp_path / "history.db"

    record_id = save_analysis(
        filename="customers.csv",
        row_count=100,
        column_count=5,
        privacy_warning_count=2,
        pii_finding_count=1,
        synthetic_row_count=80,
        db_path=db_path,
    )

    history = get_history(db_path=db_path)

    assert record_id == 1
    assert len(history) == 1
    assert history[0]["filename"] == "customers.csv"
    assert history[0]["row_count"] == 100
    assert history[0]["column_count"] == 5
    assert history[0]["privacy_warning_count"] == 2
    assert history[0]["pii_finding_count"] == 1
    assert history[0]["synthetic_row_count"] == 80
    assert history[0]["analyzed_at"]


def test_history_is_newest_first_and_respects_limit(tmp_path):
    db_path = tmp_path / "history.db"

    for filename in ("first.csv", "second.csv", "third.csv"):
        save_analysis(
            filename=filename,
            row_count=10,
            column_count=2,
            privacy_warning_count=0,
            pii_finding_count=0,
            db_path=db_path,
        )

    history = get_history(limit=2, db_path=db_path)

    assert [item["filename"] for item in history] == [
        "third.csv",
        "second.csv",
    ]


def test_synthetic_row_count_can_be_missing(tmp_path):
    db_path = tmp_path / "history.db"

    save_analysis(
        filename="source.csv",
        row_count=10,
        column_count=2,
        privacy_warning_count=1,
        pii_finding_count=0,
        db_path=db_path,
    )

    history = get_history(db_path=db_path)

    assert history[0]["synthetic_row_count"] is None


def test_rejects_negative_counts(tmp_path):
    with pytest.raises(ValueError):
        save_analysis(
            filename="data.csv",
            row_count=-1,
            column_count=2,
            privacy_warning_count=0,
            pii_finding_count=0,
            db_path=tmp_path / "history.db",
        )


def test_rejects_invalid_history_limit(tmp_path):
    with pytest.raises(ValueError):
        get_history(limit=0, db_path=tmp_path / "history.db")


def test_clear_history_deletes_all_records(tmp_path):
    db_path = tmp_path / "history.db"

    for filename in ("first.csv", "second.csv", "third.csv"):
        save_analysis(
            filename=filename,
            row_count=10,
            column_count=2,
            privacy_warning_count=1,
            pii_finding_count=0,
            db_path=db_path,
        )

    deleted_count = clear_history(db_path=db_path)

    assert deleted_count == 3
    assert get_history(db_path=db_path) == []


def test_clear_empty_history_returns_zero(tmp_path):
    db_path = tmp_path / "history.db"

    deleted_count = clear_history(db_path=db_path)

    assert deleted_count == 0
    assert get_history(db_path=db_path) == []


def test_can_save_records_after_clearing_history(tmp_path):
    db_path = tmp_path / "history.db"

    save_analysis(
        filename="old.csv",
        row_count=10,
        column_count=2,
        privacy_warning_count=0,
        pii_finding_count=0,
        db_path=db_path,
    )

    clear_history(db_path=db_path)

    save_analysis(
        filename="new.csv",
        row_count=20,
        column_count=3,
        privacy_warning_count=1,
        pii_finding_count=1,
        db_path=db_path,
    )

    history = get_history(db_path=db_path)

    assert len(history) == 1
    assert history[0]["filename"] == "new.csv"
    assert history[0]["row_count"] == 20

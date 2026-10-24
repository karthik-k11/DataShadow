
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path("instance") / "datashadow_history.db"


def init_db(db_path=DEFAULT_DB_PATH):
    """Create the local analysis history database and table."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                column_count INTEGER NOT NULL,
                privacy_warning_count INTEGER NOT NULL,
                pii_finding_count INTEGER NOT NULL,
                synthetic_row_count INTEGER,
                analyzed_at TEXT NOT NULL DEFAULT (
                    strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')
                )
            )
        """)
        connection.commit()


def save_analysis(
    filename,
    row_count,
    column_count,
    privacy_warning_count,
    pii_finding_count,
    synthetic_row_count=None,
    db_path=DEFAULT_DB_PATH,
):
    """Save summary metadata only, never the dataset contents."""
    if row_count < 0 or column_count < 0:
        raise ValueError("Row and column counts cannot be negative.")

    if privacy_warning_count < 0 or pii_finding_count < 0:
        raise ValueError("Finding counts cannot be negative.")

    if synthetic_row_count is not None and synthetic_row_count < 0:
        raise ValueError("Synthetic row count cannot be negative.")

    init_db(db_path)

    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO analysis_history (
                filename,
                row_count,
                column_count,
                privacy_warning_count,
                pii_finding_count,
                synthetic_row_count
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                row_count,
                column_count,
                privacy_warning_count,
                pii_finding_count,
                synthetic_row_count,
            ),
        )
        return cursor.lastrowid


def get_history(limit=50, db_path=DEFAULT_DB_PATH):
    """Return the newest analysis summaries first."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer.")

    init_db(db_path)

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT
                id,
                filename,
                row_count,
                column_count,
                privacy_warning_count,
                pii_finding_count,
                synthetic_row_count,
                analyzed_at
            FROM analysis_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]


def clear_history(db_path=DEFAULT_DB_PATH):
    """Delete all analysis history records without dropping the table."""
    init_db(db_path)

    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute("DELETE FROM analysis_history")
        return cursor.rowcount
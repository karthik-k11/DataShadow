
import threading
import time

import pandas as pd


DATASET_TTL_SECONDS = 30 * 60

_active_datasets = {}
_store_lock = threading.Lock()


def init_dataset_store(db_path=None):
    """Initialize the in-memory dataset store."""
    return None


def cleanup_expired_datasets(
    db_path=None,
    ttl_seconds=DATASET_TTL_SECONDS,
):
    """Remove datasets inactive beyond the TTL."""
    if ttl_seconds < 1:
        raise ValueError("TTL must be positive.")

    cutoff = time.time() - ttl_seconds

    with _store_lock:
        expired_ids = [
            dataset_id
            for dataset_id, entry in _active_datasets.items()
            if entry["last_accessed"] < cutoff
        ]

        for dataset_id in expired_ids:
            del _active_datasets[dataset_id]

    return len(expired_ids)


def store_dataset(
    dataset_id,
    filename,
    dataframe,
    db_path=None,
):
    """Keep a validated DataFrame temporarily in server memory."""
    if not dataset_id or not filename:
        raise ValueError("Dataset ID and filename are required.")

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    cleanup_expired_datasets()

    now = time.time()

    with _store_lock:
        _active_datasets[dataset_id] = {
            "filename": filename,
            "df": dataframe.copy(deep=True),
            "created_at": now,
            "last_accessed": now,
        }


def get_dataset(
    dataset_id,
    db_path=None,
    ttl_seconds=DATASET_TTL_SECONDS,
):
    """Retrieve an active dataset and refresh its inactivity timer."""
    if not dataset_id:
        return None

    cleanup_expired_datasets(ttl_seconds=ttl_seconds)

    with _store_lock:
        entry = _active_datasets.get(dataset_id)

        if entry is None:
            return None

        entry["last_accessed"] = time.time()

        return {
            "filename": entry["filename"],
            "df": entry["df"].copy(deep=True),
        }


def delete_dataset(dataset_id, db_path=None):
    """Remove a dataset from memory."""
    if not dataset_id:
        return False

    with _store_lock:
        return _active_datasets.pop(dataset_id, None) is not None
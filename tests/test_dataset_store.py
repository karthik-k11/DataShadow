
import time

import pandas as pd

from dataset_store import (
    cleanup_expired_datasets,
    delete_dataset,
    get_dataset,
    store_dataset,
)


def test_dataset_survives_store_retrieval():
    original = pd.DataFrame({
        "age": [25, 30],
        "city": ["Bengaluru", "Mysuru"],
    })

    store_dataset("session-1", "people.csv", original)
    result = get_dataset("session-1")

    assert result is not None
    assert result["filename"] == "people.csv"
    pd.testing.assert_frame_equal(result["df"], original)


def test_missing_dataset_returns_none():
    assert get_dataset("missing") is None


def test_expired_dataset_is_removed():
    store_dataset(
        "session-expiry",
        "people.csv",
        pd.DataFrame({"age": [25]}),
    )

    # Simulate inactivity beyond the expiry period.
    from dataset_store import _active_datasets

    _active_datasets["session-expiry"]["last_accessed"] = (
        time.time() - 2
    )

    assert cleanup_expired_datasets(ttl_seconds=1) >= 1
    assert get_dataset("session-expiry") is None


def test_dataset_can_be_deleted():
    store_dataset(
        "session-delete",
        "people.csv",
        pd.DataFrame({"age": [25]}),
    )

    assert delete_dataset("session-delete") is True
    assert get_dataset("session-delete") is None


def test_replacing_dataset_updates_contents():
    store_dataset(
        "session-replace",
        "first.csv",
        pd.DataFrame({"age": [25]}),
    )
    updated = pd.DataFrame({"age": [40]})
    store_dataset("session-replace", "second.csv", updated)

    result = get_dataset("session-replace")

    assert result["filename"] == "second.csv"
    pd.testing.assert_frame_equal(result["df"], updated)


def test_retrieved_dataset_is_independent_copy():
    original = pd.DataFrame({"age": [25]})
    store_dataset("session-copy", "people.csv", original)

    result = get_dataset("session-copy")
    result["df"].loc[0, "age"] = 99

    stored = get_dataset("session-copy")
    assert stored["df"].loc[0, "age"] == 25
"""
test_customer_feature_repository.py

Monkeypatches pandas.read_sql and the lazily-imported DB engine so these
tests run without a real MySQL connection, while still exercising the
real query-building / not-found / column-selection logic.
"""

from __future__ import annotations

from contextlib import contextmanager

import pandas as pd
import pytest

import app.ml.explainibility_inference.data_access.customer_feature_repository as repo
from app.ml.explainibility_inference.inference.feature_contract import ID_COLUMN, REQUIRED_FEATURES


class _FakeEngine:
    """Minimal stand-in for a SQLAlchemy engine's `.connect()` context manager."""

    @contextmanager
    def connect(self):
        yield object()


def _fake_row(customer_id: str) -> dict:
    row = {ID_COLUMN: customer_id}
    row.update({feature: 1.0 for feature in REQUIRED_FEATURES})
    return row


def test_get_customer_features_batch_returns_expected_columns(monkeypatch):
    monkeypatch.setattr(repo, "_get_engine", lambda: _FakeEngine())
    fake_result = pd.DataFrame([_fake_row("C001"), _fake_row("C002")])
    monkeypatch.setattr(pd, "read_sql", lambda *a, **k: fake_result)

    result = repo.get_customer_features_batch(["C001", "C002"])

    assert list(result.columns) == [ID_COLUMN] + REQUIRED_FEATURES
    assert result[ID_COLUMN].tolist() == ["C001", "C002"]


def test_get_customer_features_single_customer(monkeypatch):
    monkeypatch.setattr(repo, "_get_engine", lambda: _FakeEngine())
    fake_result = pd.DataFrame([_fake_row("C001")])
    monkeypatch.setattr(pd, "read_sql", lambda *a, **k: fake_result)

    result = repo.get_customer_features("C001")

    assert len(result) == 1
    assert result.iloc[0][ID_COLUMN] == "C001"


def test_no_matching_rows_raises_customer_not_found(monkeypatch):
    monkeypatch.setattr(repo, "_get_engine", lambda: _FakeEngine())
    monkeypatch.setattr(pd, "read_sql", lambda *a, **k: pd.DataFrame())

    with pytest.raises(repo.CustomerNotFoundError):
        repo.get_customer_features("UNKNOWN")


def test_empty_id_list_raises_value_error():
    with pytest.raises(ValueError):
        repo.get_customer_features_batch([])

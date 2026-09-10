import importlib
from unittest.mock import Mock
import pandas as pd
import app.Database.ingest as ingest

def test_resolve_csv_path_prefers_original_file(tmp_path):
    dataset_dir = str(tmp_path)
    original = tmp_path / "data.csv"
    original.write_text("id,name\n1,A\n", encoding="utf-8")
    result = ingest._resolve_csv_path(dataset_dir, "data.csv")
    assert result == str(original)

def test_resolve_csv_path_uses_clean_file_when_original_missing(tmp_path):
    dataset_dir = str(tmp_path)
    clean_file = tmp_path / "data_clean.csv"
    clean_file.write_text("id,name\n1,A\n", encoding="utf-8")
    result = ingest._resolve_csv_path(dataset_dir, "data.csv")
    assert result == str(clean_file)

def test_resolve_csv_path_returns_primary_path_when_file_missing(tmp_path):
    dataset_dir = str(tmp_path)
    result = ingest._resolve_csv_path(dataset_dir, "data.csv")
    assert result == str(tmp_path / "data.csv")

def test_print_before_cleaning_summary(capsys):
    df = pd.DataFrame({
        "id": [1, 2, 2],
        "name": ["A", None, "C"],
    })
    ingest.print_before_cleaning_summary("customers", df)
    output = capsys.readouterr().out
    assert "CUSTOMERS - BEFORE CLEANING" in output
    assert "Rows:" in output
    assert "Columns:" in output
    assert "Missing values:" in output
    assert "Duplicate full rows:" in output


def test_print_after_cleaning_summary(capsys):
    before_df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["A", None, "C"],
    })
    after_df = pd.DataFrame({
        "id": [1, 3],
        "name": ["A", "C"],
    })
    ingest.print_after_cleaning_summary(
        "customers",
        before_df,
        after_df,
    )
    output = capsys.readouterr().out
    assert "CUSTOMERS - AFTER CLEANING" in output
    assert "Rows before:          3" in output
    assert "Rows after:           2" in output
    assert "Rows removed:         1" in output
    assert "Missing before:" in output
    assert "Missing after:" in output

def test_print_validation_summary_with_primary_key(capsys):
    df = pd.DataFrame({
        "customer_id": ["c1", "c2"],
        "name": ["A", "B"],
    })
    ingest.print_validation_summary("customers", df)
    output = capsys.readouterr().out
    assert "CUSTOMERS - VALIDATION" in output
    assert "Rows ready for MySQL: 2" in output
    assert "Remaining NULLs:      0" in output
    assert "Primary key:          customer_id" in output
    assert "Missing primary keys: 0" in output
    assert "Duplicate primary keys: 0" in output
    assert "Primary key validation: PASSED" in output

def test_print_validation_summary_without_primary_key(capsys):
    df = pd.DataFrame({
        "payment_value": [100, 200],
    })
    ingest.print_validation_summary("order_payments", df)
    output = capsys.readouterr().out
    assert "ORDER_PAYMENTS - VALIDATION" in output
    assert "Rows ready for MySQL: 2" in output
    assert "Primary key:" not in output

def test_clear_tables_in_fk_safe_order(monkeypatch):
    executed = []
    fake_connection = Mock()
    fake_connection.execute.side_effect = (
        lambda statement: executed.append(str(statement))
    )
    fake_inspector = Mock()
    fake_inspector.has_table.return_value = True
    fake_engine = Mock()
    transaction = Mock()
    transaction.__enter__ = Mock(return_value=fake_connection)
    transaction.__exit__ = Mock(return_value=False)
    fake_engine.begin.return_value = transaction
    monkeypatch.setattr(ingest, "engine", fake_engine)
    monkeypatch.setattr(ingest, "inspect", Mock(return_value=fake_inspector))
    ingest.clear_tables(["customers", "orders", "order_items"])

    assert executed == [
        "DELETE FROM users",
        "DELETE FROM order_items",
        "DELETE FROM orders",
        "DELETE FROM customers",
    ]

def test_load_table_reads_cleans_and_loads(monkeypatch, tmp_path):
    csv_path = tmp_path / "olist_customers_dataset.csv"
    csv_path.write_text("customer_id,name\nc1,A\n", encoding="utf-8")
    source_df = pd.DataFrame({
        "customer_id": ["c1"],
        "name": ["A"],
    })
    cleaned_df = pd.DataFrame({
        "customer_id": ["c1"],
        "name": ["A"],
    })
    monkeypatch.setattr(ingest, "DATASET_DIR", str(tmp_path))
    monkeypatch.setattr(
        ingest.pd,
        "read_csv",
        Mock(return_value=source_df),
    )
    monkeypatch.setattr(
        ingest,
        "clean_dataframe",
        Mock(return_value=cleaned_df),
    )
    to_sql_mock = Mock()
    monkeypatch.setattr(pd.DataFrame, "to_sql", to_sql_mock)
    valid_keys = {}
    ingest.load_table("customers", valid_keys)
    ingest.clean_dataframe.assert_called_once_with(
        "customers",
        source_df,
        valid_keys=valid_keys,
    )
    to_sql_mock.assert_called_once()
    assert valid_keys["customers"] == {"c1"}

def test_load_table_skips_empty_cleaned_dataframe(monkeypatch, tmp_path):
    csv_path = tmp_path / "olist_customers_dataset.csv"
    csv_path.write_text("customer_id,name\nc1,A\n", encoding="utf-8")
    source_df = pd.DataFrame({
        "customer_id": ["c1"],
        "name": ["A"],
    })
    cleaned_df = pd.DataFrame(columns=["customer_id", "name"])
    monkeypatch.setattr(ingest, "DATASET_DIR", str(tmp_path))
    monkeypatch.setattr(
        ingest.pd,
        "read_csv",
        Mock(return_value=source_df),
    )
    monkeypatch.setattr(
        ingest,
        "clean_dataframe",
        Mock(return_value=cleaned_df),
    )
    to_sql_mock = Mock()
    monkeypatch.setattr(pd.DataFrame, "to_sql", to_sql_mock)
    valid_keys = {}
    ingest.load_table("customers", valid_keys)
    to_sql_mock.assert_not_called()
    assert valid_keys["customers"] == set()

def test_load_table_geolocation_uses_chunks(monkeypatch, tmp_path):
    csv_path = tmp_path / "olist_geolocation_dataset.csv"
    csv_path.write_text("zip,lat,lng\n100,17,78\n", encoding="utf-8")
    chunk1 = pd.DataFrame({
        "geolocation_zip_code_prefix": [100],
        "geolocation_lat": [17.0],
        "geolocation_lng": [78.0],
    })
    chunk2 = pd.DataFrame({
        "geolocation_zip_code_prefix": [200],
        "geolocation_lat": [18.0],
        "geolocation_lng": [79.0],
    })
    fake_reader = iter([chunk1, chunk2])
    read_csv_mock = Mock(return_value=fake_reader)
    monkeypatch.setattr(ingest, "DATASET_DIR", str(tmp_path))
    monkeypatch.setattr(ingest.pd, "read_csv", read_csv_mock)
    clean_mock = Mock(side_effect=[chunk1, chunk2])
    monkeypatch.setattr(ingest, "clean_dataframe", clean_mock)
    to_sql_mock = Mock()
    monkeypatch.setattr(pd.DataFrame, "to_sql", to_sql_mock)
    valid_keys = {}
    ingest.load_table("geolocation", valid_keys)
    read_csv_mock.assert_called_once_with(
        str(csv_path),
        chunksize=10000,
    )

    assert clean_mock.call_count == 2
    assert to_sql_mock.call_count == 2

import importlib
import sys
from unittest.mock import Mock
import pytest
MODULE = "app.Database.database"

def _remove_database_module():
    """Remove database.py from the import cache so each test starts fresh."""
    sys.modules.pop(MODULE, None)

def test_missing_required_environment_variables(monkeypatch):
    """database.py should fail fast when required DB settings are missing."""
    required_vars = [
        "DB_HOST",
        "DB_PORT",
        "DB_USER",
        "DB_PASSWORD",
        "DB_NAME",
    ]
    for variable in required_vars:
        monkeypatch.delenv(variable, raising=False)
    # Prevent database.py from loading values from the project's .env file.
    monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
    _remove_database_module()
    with pytest.raises(RuntimeError, match="Missing required environment"):
        importlib.import_module(MODULE)

def test_database_configuration(monkeypatch):
    """database.py should create the expected MySQL SQLAlchemy configuration."""
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PORT", "3306")
    monkeypatch.setenv("DB_USER", "test_user")
    monkeypatch.setenv("DB_PASSWORD", "test_password")
    monkeypatch.setenv("DB_NAME", "test_db")
    # Prevent values from the project's .env from affecting this test.
    monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
    fake_engine = Mock()
    import sqlalchemy
    monkeypatch.setattr(
        sqlalchemy,
        "create_engine",
        Mock(return_value=fake_engine),
    )
    _remove_database_module()
    database = importlib.import_module(MODULE)
    assert database.DB_HOST == "localhost"
    assert database.DB_PORT == "3306"
    assert database.DB_USER == "test_user"
    assert database.DB_PASSWORD == "test_password"
    assert database.DB_NAME == "test_db"
    sqlalchemy.create_engine.assert_called_once()
    args, kwargs = sqlalchemy.create_engine.call_args
    assert args[0].drivername == "mysql+pymysql"
    assert args[0].username == "test_user"
    assert args[0].password == "test_password"
    assert args[0].host == "localhost"
    assert args[0].port == 3306
    assert args[0].database == "test_db"
    assert kwargs["pool_pre_ping"] is True
    assert kwargs["pool_recycle"] == 3600
    assert database.engine is fake_engine

def test_missing_port_is_rejected(monkeypatch):
    """DB_PORT is required by database.py and should not be omitted."""
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.delenv("DB_PORT", raising=False)
    monkeypatch.setenv("DB_USER", "test_user")
    monkeypatch.setenv("DB_PASSWORD", "test_password")
    monkeypatch.setenv("DB_NAME", "test_db")
    # Prevent database.py from loading values from the project's .env file.
    monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
    _remove_database_module()
    with pytest.raises(RuntimeError, match="DB_PORT"):
        importlib.import_module(MODULE)

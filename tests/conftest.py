import sys
import importlib

import pytest


@pytest.fixture()
def app_db(tmp_path, monkeypatch):
    """
    Provide (app_module, db_module) using an isolated SQLite DB per test.

    This avoids polluting the developer DB at `data/app.db`.
    """
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("MELX_HEALTH_DB_PATH", str(db_path))
    monkeypatch.setenv("CHATBOT_PROVIDER", "local")
    monkeypatch.setenv("CHATBOT_NO_PHI", "1")

    sys.modules.pop("db", None)
    sys.modules.pop("app", None)

    db_module = importlib.import_module("db")
    db_module.init_db()
    app_module = importlib.import_module("app")
    app_module.app.config["TESTING"] = True
    return app_module, db_module

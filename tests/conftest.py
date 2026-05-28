import sqlite3
import pytest
from email_rag.db import Database


@pytest.fixture
def db(tmp_path):
    database = Database(str(tmp_path / "test.db"))
    database.init_schema(dim=4)  # tiny dim for fast tests
    yield database
    database.close()

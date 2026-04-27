"""DuckDB connection helper.

Returns a connection with the schema applied. No pooling, no ORM.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import duckdb
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_conn(path: str | None = None) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection with the schema applied.

    *path* overrides ``DUCKDB_PATH`` from env.  Pass ``":memory:"`` for tests.
    """
    if path is None:
        load_dotenv()
        path = os.environ.get("DUCKDB_PATH", "./data/edgeful.duckdb")

    conn = duckdb.connect(path)
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Apply schema.sql if the core tables don't exist yet."""
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
    }
    if "bars_1min" not in tables:
        logger.info("Applying schema from %s", _SCHEMA_PATH)
        sql = _SCHEMA_PATH.read_text()
        conn.execute(sql)

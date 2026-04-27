"""Unit tests for data.db — uses in-memory DuckDB, no disk I/O."""

from __future__ import annotations

import duckdb
import pytest

from data.db import get_conn


def _table_names(conn: duckdb.DuckDBPyConnection) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
    }


class TestGetConn:
    def test_creates_tables_on_first_call(self) -> None:
        conn = get_conn(":memory:")
        tables = _table_names(conn)
        assert "bars_1min" in tables
        assert "bars_daily" in tables
        assert "backfill_log" in tables
        assert "data_quality" in tables
        conn.close()

    def test_idempotent_schema(self) -> None:
        conn = get_conn(":memory:")
        # Insert a row so we can verify it survives a second schema apply
        conn.execute(
            "INSERT INTO bars_1min VALUES ('NIFTY', '2025-01-06 09:15:00', "
            "21800.0, 21810.0, 21790.0, 21805.0, 1000)"
        )
        # Re-apply schema (simulates restart)
        from data.db import _ensure_schema

        _ensure_schema(conn)

        count: int = conn.execute("SELECT count(*) FROM bars_1min").fetchone()[0]  # type: ignore[index]
        assert count == 1
        conn.close()

    def test_bars_1min_primary_key(self) -> None:
        conn = get_conn(":memory:")
        row = (
            "INSERT INTO bars_1min VALUES "
            "('NIFTY', '2025-01-06 09:15:00', 21800.0, 21810.0, 21790.0, 21805.0, 1000)"
        )
        conn.execute(row)
        with pytest.raises(duckdb.ConstraintException):
            conn.execute(row)
        conn.close()

    def test_backfill_log_table_exists(self) -> None:
        conn = get_conn(":memory:")
        conn.execute(
            "INSERT INTO backfill_log VALUES ('NIFTY', '2025-01', '2025-06-01 12:00:00', 7500)"
        )
        rows = conn.execute("SELECT * FROM backfill_log").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "NIFTY"
        assert rows[0][1] == "2025-01"
        conn.close()

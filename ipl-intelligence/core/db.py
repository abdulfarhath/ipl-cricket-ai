"""Connection pooling + tiny query helpers (sync pool; FastAPI runs it in a
threadpool via def endpoints, MCP/agent call it directly)."""
import logging

from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

from core.config import DATABASE_URL

log = logging.getLogger("db")
_pool: ConnectionPool | None = None


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(DATABASE_URL, min_size=1, max_size=10,
                               kwargs={"row_factory": dict_row}, open=True)
    return _pool


def _clean(v):
    """JSON-safe values: Decimal -> int/float, date -> ISO string."""
    import datetime
    from decimal import Decimal

    if isinstance(v, Decimal):
        return int(v) if v == v.to_integral_value() else float(v)
    if isinstance(v, (datetime.date, datetime.datetime)):
        return v.isoformat()
    return v


def q(sql: str, params: tuple = ()) -> list[dict]:
    with pool().connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [{k: _clean(v) for k, v in r.items()} for r in rows]


def q1(sql: str, params: tuple = ()) -> dict | None:
    rows = q(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params: tuple = ()) -> None:
    with pool().connection() as conn:
        conn.execute(sql, params)

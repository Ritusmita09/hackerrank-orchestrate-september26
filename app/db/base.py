"""Database engine/session management.

No engine is created at import time: importing the app (and running the
Phase 1 tests) must not require a database server. Engines are built lazily
from ``Settings.database_url`` — PostgreSQL in production, SQLite for local
development and tests.
"""
from __future__ import annotations

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ..config import get_settings


def build_engine(url: str | None = None, *, echo: bool = False) -> Engine:
    """Create a SQLAlchemy engine for ``url`` (defaults to the configured URL)."""
    url = url or get_settings().database_url
    engine = create_engine(url, echo=echo, future=True)
    if url.startswith("sqlite"):
        # SQLite ignores NVARCHAR lengths and foregoes FK enforcement by
        # default; turning enforcement on keeps local/test parity with
        # PostgreSQL semantics.
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, _record):  # pragma: no cover
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    return engine


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False,
                        future=True)

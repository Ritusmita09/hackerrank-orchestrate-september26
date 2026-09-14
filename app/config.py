"""Application configuration.

All knobs come from environment variables — never hardcoded. A local ``.env``
file is loaded if present (never committed; see ``.env.example`` for the
documented keys). Secrets are only ever read from the environment.

No database connection is made at import time: importing this module (and the
whole app) works with no PostgreSQL server running. SQLite is the
development/test fallback; PostgreSQL is the production target.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Tuple

try:  # optional; the app also runs with plain environment variables
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is in requirements
    load_dotenv = None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"environment variable {name} must be an integer, got {raw!r}") from exc


def _env_tuple(name: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    """Every runtime knob of the backend, sourced from the environment."""

    # -- environment ---------------------------------------------------------
    app_env: str = "local"                      # local | test | staging | production
    log_level: str = "INFO"
    log_json: bool = True

    # -- database --------------------------------------------------------------
    #: PostgreSQL in production (postgresql+psycopg://...), SQLite fallback
    #: for local development and the test suite (sqlite+pysqlite:///...).
    database_url: str = "sqlite:///./ai_financial_agent.db"

    # -- AI layer (Phase 3+; wired but unused in Phase 2) ----------------------
    llm_provider: str = "none"                  # none | omniroute | (future adapters)
    llm_model: str = ""
    anthropic_api_key: Optional[str] = None
    llm_max_steps_per_turn: int = 5
    llm_max_tokens_per_turn: int = 4000

    # -- OmniRoute routing layer (Phase 4) --------------------------------------
    #: Local model-routing endpoint speaking the Anthropic Messages API,
    #: e.g. http://localhost:20128. Credentials come from the environment only.
    omniroute_base_url: str = ""
    omniroute_api_key: Optional[str] = None
    #: Provider-prefixed model id, e.g. "agentrouter/deepseek-v4-flash".
    omniroute_model: str = ""
    #: Extraction turn model (multimodal). Falls back to omniroute_model.
    omniroute_extraction_model: str = ""

    # -- ingestion safety -------------------------------------------------------
    max_upload_bytes: int = 5 * 1024 * 1024     # 5 MB hard cap for uploads
    allowed_mime_types: Tuple[str, ...] = (
        "text/csv",
        "application/pdf",
        "image/png",
        "image/jpeg",
        "text/plain",
    )
    max_message_chars: int = 10_000
    max_csv_rows: int = 10_000
    #: Root directory for uploaded document bytes (documents table stores metadata only).
    storage_dir: str = "storage"

    # -- forecasting defaults ----------------------------------------------------
    default_horizon_days: int = 90


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build the singleton Settings from the environment (cached)."""
    return Settings(
        app_env=os.environ.get("APP_ENV", "local"),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        log_json=_env_bool("LOG_JSON", True),
        database_url=os.environ.get(
            "DATABASE_URL", "sqlite:///./ai_financial_agent.db"
        ),
        llm_provider=os.environ.get("LLM_PROVIDER", "none"),
        llm_model=os.environ.get("LLM_MODEL", ""),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        llm_max_steps_per_turn=_env_int("LLM_MAX_STEPS_PER_TURN", 5),
        llm_max_tokens_per_turn=_env_int("LLM_MAX_TOKENS_PER_TURN", 4000),
        omniroute_base_url=os.environ.get("OMNIROUTE_BASE_URL", ""),
        omniroute_api_key=os.environ.get("OMNIROUTE_API_KEY"),
        omniroute_model=os.environ.get("OMNIROUTE_MODEL", ""),
        omniroute_extraction_model=os.environ.get("OMNIROUTE_EXTRACTION_MODEL", ""),
        max_upload_bytes=_env_int("MAX_UPLOAD_BYTES", 5 * 1024 * 1024),
        allowed_mime_types=_env_tuple(
            "ALLOWED_MIME_TYPES",
            ("text/csv", "application/pdf", "image/png", "image/jpeg", "text/plain"),
        ),
        max_message_chars=_env_int("MAX_MESSAGE_CHARS", 10_000),
        max_csv_rows=_env_int("MAX_CSV_ROWS", 10_000),
        storage_dir=os.environ.get("STORAGE_DIR", "storage"),
        default_horizon_days=_env_int("DEFAULT_HORIZON_DAYS", 90),
    )

# Backend Design — Phase 2

This document describes the backend infrastructure built in Phase 2: the
request flow, the database model, the service layer, the API boundary, how
the locked Phase 1 engine is invoked, and how everything is auditable.

The deterministic financial engine (`app/core/`, Phase 1) is **locked** — the
backend wraps it; it never modifies or weakens it.

## Overview

```
HTTP client
   │
   ▼
FastAPI app (app/api/main.py)          ← middleware: request IDs, error mapping
   │  routes (app/api/routes/)          ← parse/validate only, no business logic
   ▼
Service layer (app/services/)          ← all business logic, DB transactions
   │  state reconstruction (financial_state_service)
   ▼
Phase 1 engine (app/core/)             ← pure, deterministic, locked
   │
   ▼
PostgreSQL (production) / SQLite (dev/test) via SQLAlchemy 2.x + Alembic
```

## Request flow

1. **Middleware** assigns/reuses an `X-Request-ID`, stored in a `ContextVar`
   for log correlation and echoed on the response.
2. **Route** parses and validates input through Pydantic schemas
   (`app/api/schemas.py`). SQLAlchemy ORM models are never exposed across
   the API boundary.
3. **Route** calls a service function with a per-request `Session`
   (`get_db` dependency, lazily initialized engine) and commits on success.
4. **Service** loads/reconstructs state, invokes the engine, persists
   results and audit events.
5. **Errors**: domain service errors (`UserServiceError`,
   `TransactionServiceError`, `InputServiceError`, `EngineError`) are mapped
   by exception handlers in `app/api/main.py` to structured JSON with proper
   status codes. Unexpected exceptions become a generic
   `500 {"error": "internal_server_error"}` — raw stack traces never leak.

## Database model (`app/db/models.py`)

Append-oriented: financial history is never destructively updated. Every
decision is traceable to a user, an input-state hash, an engine version,
and timestamps.

| Table | Purpose |
|---|---|
| `users` | Identity, home currency (single currency per user) |
| `financial_profiles` | Versioned balances/preferences (revision per row; latest active row wins) |
| `accounts` | Registered accounts (metadata only; no bank APIs in Phase 2) |
| `transactions` | Append-mostly ledger of financial events, with amendments |
| `recurring_patterns` | Confirmed recurring income/expense patterns fed to the engine |
| `affordability_requests` + `payment_options` | Persisted user purchase requests and their payment options |
| `financial_goals` / `user_preferences` / `financial_facts` | User-stated goals, preferences, confirmed facts |
| `documents` / `document_extractions` | Uploaded documents (content-hashed) and draft extractions (Phase 3: LLM) |
| `messages` | Raw user messages (Phase 3: LLM interpretation; stored as drafts) |
| `decisions` / `verification_results` | Append-only decision records: status, selected plan, fact sheet, input hash, engine version |
| `forecast_runs` | Persisted simulation series (replayable via engine_version + input_hash) |
| `audit_events` | Append-only audit trail of every consequential action |

### Migrations

Alembic (`alembic/`) is the **authoritative** schema migration mechanism —
the application never calls `Base.metadata.create_all()` at startup; a
deployment must run migrations before the API serves requests. The URL is
injected at runtime from `app.config.get_settings()` (env var
`DATABASE_URL` / `.env`) — `alembic.ini` holds only a placeholder, so
**importing or migrating the app never requires a real PostgreSQL server**.

```bash
# Local development: migrate, then start the API
python -m alembic upgrade head
python -m uvicorn app.api.main:create_app --factory --reload

# Inspect state
python -m alembic current          # which revision the DB is at
python -m alembic history          # migration history

# New migration after model changes
python -m alembic revision --autogenerate -m "..."
```

`upgrade head` is idempotent: re-running it against an already-migrated
database is a no-op, so it is safe in every startup path.

**Deployments** use `entrypoint.sh`, which runs the exact same sequence
(`alembic upgrade head`, then uvicorn) before accepting traffic — a fresh
container can never serve against a missing or stale schema. The Dockerfile
uses it as its ENTRYPOINT.

**Tests** are unaffected: they use in-memory SQLite and create their schema
in the test fixture (`app/tests/test_api.py`), which is test-only isolation,
not an application-startup path. Phase 1 engine tests (`app/core`) never
touch a database.

## Service layer (`app/services/`)

All financial business logic lives here — never in route functions.

- **`user_service`** — user + profile CRUD, DB profile → engine
  `UserFinancialProfile` translation.
- **`transaction_service`** — manual ingestion with stream-key dedup
  (SHA-256 over date/direction/amount/currency/normalized description),
  DB row → engine `FinancialEvent` translation.
- **`ingestion_service`** — secure CSV import: column validation, ISO dates,
  currency must equal home currency, amounts as integer minor units or
  ≤2-decimal values, per-row error reporting, row-count bound.
- **`document_service`** — document registration (size/MIME limits,
  content-hash, no content in audit payloads) and message intake limits.
  Extractions are drafts until the user confirms — untrusted input never
  reaches the ledger unconfirmed.
- **`pattern_service`** — DB recurring patterns → engine
  `RecurringPattern` (fixed-day / monthly-day periods).
- **`financial_state_service`** — **state reconstruction**: latest profile +
  all transaction rows + confirmed patterns → engine's `build_state()`.
- **`forecast_service`** — simulation over reconstructed state; persists the
  series (replayable), audits the run.
- **`decision_service`** — the decision pipeline (below).
- **`audit_service`** — append-only audit events.

## Engine invocation

`decision_service.analyze_request` runs the locked Phase 1 pipeline:

1. `generate_candidate_plans(request, state, assumptions)`
2. `verify_plan(plan, ...)` — independent simulation gate for **every**
   candidate; failures are counted, never returned
3. `rank_plans(survivors, request)` — strict lexicographic ordering
4. Fact sheet compiled from verification evidence + safe-payment bounds

All money is integer minor units; the engine refuses floats and mixed
currencies. Results carry `ENGINE_VERSION` and a SHA-256 `input_hash`.

## API boundary

Endpoints (all under the FastAPI app from `app.api.main:create_app`):

| Method & path | Purpose |
|---|---|
| `GET /health` | Liveness + engine version |
| `POST /users`, `GET /users/{user_id}` | User lifecycle |
| `GET /users/{user_id}/financial-state` | Fully reconstructed deterministic state |
| `GET /users/{user_id}/financial-state/profile` | Latest profile revision |
| `POST /users/{user_id}/transactions`, `GET .../transactions` | Manual ingestion / listing |
| `POST /users/{user_id}/transactions/csv` | Batch CSV import with row-level errors |
| `POST /users/{user_id}/documents` | Document upload (metadata registration) |
| `POST /users/{user_id}/simulate` | Deterministic forecast over current state |
| `POST /users/{user_id}/decision` | Affordability decision (no LLM in Phase 2) |

## Auditability

Every consequential action writes an `audit_events` row (event type, user,
request id, small non-sensitive payload). Decisions additionally persist the
selected plan, fact sheet, verification verdict, input-state hash, and
engine version — a stored decision can be replayed and explained.

## Configuration & security

`app/config.py` reads environment variables (with optional `.env` via
python-dotenv; see `.env.example`; **no real secrets are committed**):
`DATABASE_URL`, `ENVIRONMENT`, `LOG_LEVEL`, `LLM_PROVIDER`/`LLM_MODEL`
(reserved for Phase 3), upload limits, MIME allowlist, CSV row bound.

Security measures: file size/MIME limits, Pydantic input validation,
parameterized queries via SQLAlchemy, request IDs for traceability,
structured error responses (no exception leakage), no secret logging,
minimal audit payloads.

## PostgreSQL in production, SQLite in development

- `DATABASE_URL=postgresql+psycopg://user:pass@host/db` in production.
- Default (unset) is `sqlite:///./ai_financial_agent.db` so the app runs
  locally with zero setup; tests use in-memory SQLite. The SQLite engine
  enables `PRAGMA foreign_keys = ON` so FK integrity matches PostgreSQL.

## Docker

`Dockerfile` runs `entrypoint.sh` (migrate → serve) via uvicorn;
`docker-compose.yml` provides `api` + `postgres` for a one-command
production-like environment:

```bash
docker compose up --build     # postgres boots first (healthcheck-gated),
                              # api migrates then serves on :8000
```

`.dockerignore` keeps secrets (`.env`) and local artifacts out of the image.

> **Verification status:** the compose file is structurally validated (YAML,
> service graph, healthcheck-gated dependency) and the entrypoint's exact
> command sequence was verified end-to-end against SQLite locally. The
> container image itself has **not** been built on this machine — Docker is
> not installed — so the Docker build/run verification is **deferred** until
> Docker is available.

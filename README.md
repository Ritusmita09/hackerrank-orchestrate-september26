# AI Financial Agent

An AI-powered financial decision-making agent that helps users make safe, informed financial choices through deterministic planning, verification, and natural language interaction.

## Project Description

The AI Financial Agent is a backend service that combines a deterministic financial engine with an LLM-driven reasoning loop to provide verifiable, safe financial advice. Users can ask questions about affordability, planning, and financial state, and the agent responds with recommendations that are mathematically proven to satisfy safety constraints.

## Problem Statement

Individuals face complex financial decisions involving trade-offs between spending, saving, and borrowing. Existing tools often rely on opaque calculations or give advice without guarantees of safety. This project provides a transparent system where every recommendation is backed by a verifiable simulation, ensuring that suggested actions never violate user-defined safety constraints (e.g., minimum balance requirements).

## Key Capabilities

- **Deterministic Financial Planning**: Core financial engine uses pure integer arithmetic, guaranteeing reproducible results.
- **LLM-Powered Reasoning**: Natural language interface allows users to interact via conversation.
- **Verification Gate**: Every candidate plan is re-simulated against the user's financial state to enforce safety constraints.
- **Memory System**: Persistent storage of user preferences, goals, and facts that inform the agent's context.
- **Secure Data Ingestion**: CSV, document, and message processing with user confirmation before data affects the ledger.
- **Auditability**: All decisions, verifications, and forecasts are stored with input hashes and engine versions for replay and review.
- **Extensible Design**: Provider‑swappable LLM abstraction and clean separation between deterministic engine and agent loop.

## Architecture Overview

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed diagram and explanation. The system consists of:

- **FastAPI Backend** – HTTP API exposing chat, ingestion, simulation, and decision endpoints.
- **Agent Orchestrator** – Bounded reasoning loop that calls tools based on LLM output.
- **Deterministic Financial Engine** – Pure‑functions core (forecast, affordability, planning, verification) with zero LLM dependencies.
- **Tool Layer** – Registered functions for state, transactions, documents, messages, simulation, verification, and memory.
- **Persistence Layer** – SQLAlchemy models backed by PostgreSQL (production) or SQLite (development).
- **Observability** – Structured logging, tracing, usage/cost accounting, and audit trails.

## Main Components

- `app/api/` – FastAPI routes and schemas.
- `app/agent/` – Orchestrator, LLM provider abstraction, prompt/context construction.
- `app/core/` – Deterministic financial engine (models, forecast, planner, verifier, etc.).
- `app/tools/` – Domain‑specific tools (state, transaction, document, message, simulation, verification, memory).
- `app/services/` – Business logic layer coordinating DB, engine, and external effects.
- `app/db/` – SQLAlchemy models, Alembic migrations.
- `app/memory/` – Preference, goal, fact, and conversation models.
- `app/observability/` – Logging, tracing, usage.
- `app/security/` – Input guards and credential detection.

## Data Flow & Decision Flow

1. **User Interaction**: User sends a message via `/api/chat` (or frontend).
2. **Context Building**: Orchestrator assembles conversation history, financial summary, and memory.
3. **LLM Reasoning**: LLM proposes tool calls (e.g., `compute_safe_amount`, `generate_candidate_plans`).
4. **Tool Execution**: Registry validates arguments, executes tools, returns results.
5. **Iteration**: Loop continues until LLM provides a final answer.
6. **Verification**: Any financial recommendation references a plan ID that has passed the verification gate.
7. **Response**: Final answer is returned to the user, with usage and tracing recorded.

## Input and Output

- **Inputs**: Natural language messages, CSV files, document uploads, manual transactions, user‑configured preferences/goals/facts.
- **Outputs**: 
  - Chat responses (plain text, possibly with tool‑generated data like balances or plans).
  - API responses (JSON) for state, transactions, plans, verifications, etc.
  - Persistent effects: new transactions, updated preferences, stored decisions, audit events.

## Safety and Security Approach

- **Credential Guard**: The memory layer actively rejects attempts to store secrets (API keys, passwords, tokens) regardless of key or value patterns ([SECURITY_AND_SAFETY.md](SECURITY_AND_SAFETY.md), [app/memory/secrets.py](app/memory/secrets.py)).
- **Deterministic Engine**: Guarantees that safety verification is reproducible and not subject to LLM variability.
- **Verification Gate**: API serializers refuse to return a plan unless its `VerificationResult.passed == True`.
- **Input Validation**: All API endpoints use Pydantic schemas; SQLAlchemy ORM prevents injection.
- **Audit Trail**: Decisions, verifications, forecast runs, and audit events are append‑only and include input hashes and engine versions.
- **Environment Configuration**: Secrets are expected only via environment variables or secret managers; `.gitignore` excludes `.env`, caches, and local artifacts.

## Testing

- **Test Suite**: 237 tests pass, 0 failures, 1 intentional skip (live OmniRoute test).
- **Run Tests**: `python run_tests.py` (discovers and executes `app/tests/test_*.py`).
- **Test Scope**: Unit, property, agent, adversarial, and safety tests covering engine, tools, services, and API endpoints.
- **Offline Guarantee**: Memory test suite enforces that no network calls are made (socket creation raises AssertionError).

## Installation & Setup

1. **Clone the repository** (already done).
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Environment Variables**: Copy `.env.example` to `.env` and fill in required values (e.g., `DATABASE_URL`). No real secrets should be committed.
4. **Database Setup**:
   - **Development (SQLite)**: No setup needed; the app creates a local file by default.
   - **PostgreSQL**: Create a database and set `DATABASE_URL=postgresql+psycopg://user:pass@host/db`.
5. **Apply Migrations**:
   ```bash
   python -m alembic upgrade head
   ```
   (Required before first run; handled by `entrypoint.sh` in Docker.)

## How to Run

- **Directly (development)**:
  ```bash
  # Ensure migrations are applied
  python -m alembic upgrade head
  # Start the API
  python -m uvicorn app.api.main:create_app --factory --reload
  ```
  The API will be available at `http://localhost:8000`.

- **Using Docker** (if Docker is installed):
  ```bash
  docker compose up --build
  ```
  This starts the API, PostgreSQL, and (when added) a frontend service.

## Project Structure

```
ai-financial-agent/
├── app/                     # Main application package
│   ├── api/                 # FastAPI routes, schemas, middleware
│   ├── agent/               # Orchestrator, LLM provider abstraction
│   ├── core/                # Deterministic financial engine (locked)
│   ├── tools/               # Registered tools (state, transaction, document, etc.)
│   ├── services/            # Business logic layer (user, transaction, ingestion, etc.)
│   ├── db/                  # SQLAlchemy models, Alembic migrations
│   ├── memory/              # Preference, goal, fact, conversation models
│   ├── observability/       # Logging, tracing, usage
│   └── security/            # Input guards, credential detection
├── alembic/                 # Database migration scripts
├── tests/                   # Test suite (mirrors app/ structure)
├── .gitignore               # Excludes secrets, caches, storage, build artifacts
├── .env.example             # Template for environment variables
├── docker-compose.yml       # Defines api, db, and (optional) frontend services
├── Dockerfile               # Entrypoint runs migrations then serves via uvicorn
├── entrypoint.sh            # Runs alembic upgrade head then uvicorn
├── requirements.txt         # Python dependencies
├── run_tests.py             # Discovers and runs unittest suite
├── ARCHITECTURE.md          # Detailed system architecture and design
├── BACKEND_DESIGN.md        # Backend infrastructure, request flow, DB model
├── PROJECT_SPEC.md          # Original project specification
├── ROADMAP.md               # Planned milestones and future work
├── SECURITY_AND_SAFETY.md   # Security policy and credential‑guarding details
└── README.md                # This file
```

## Current Limitations / Known Limitations

- **No Frontend Yet**: The repository contains only the backend API. A Streamlit‑based frontend is referenced in the architecture documents but is not present in this release.
- **Single‑Currency MVP**: The current implementation assumes a single home currency per user; multi‑currency support is planned for future work.
- **Basic Authentication**: The MVP uses simple session‑based login; production‑grade authentication (OAuth2, API keys, etc.) is to be added according to `SECURITY_AND_SAFETY.md`.
- **Document Processing**: While the ingestion pipeline accepts files, the actual LLM‑based extraction and confirmation steps rely on tool implementations that are present but may require further validation for production use.
- **Rate Limiting & Cost Control**: The agent loop has per‑turn step/token budgets, but global rate limiting and cost caps are configuration‑driven and should be tuned for deployment.

## Documentation References

- [ARCHITECTURE.md](ARCHITECTURE.md) – Full system design, diagrams, and component responsibilities.
- [BACKEND_DESIGN.md](BACKEND_DESIGN.md) – Backend‑specific details: request flow, service layer, database model, auditability.
- [PROJECT_SPEC.md](PROJECT_SPEC.md) – Original project specification and goals.
- [ROADMAP.md](ROADMAP.md) – Planned features and milestones.
- [SECURITY_AND_SAFETY.md](SECURITY_AND_SAFETY.md) – Security policy, credential‑guarding rules, and authentication guidance.

## Current Project Status

- **Backend**: Fully implemented, tested, and ready for deployment.
- **Tests**: 237 passing, 0 failing, 1 skipped (intentional live OmniRoute test).
- **Documentation**: All key design documents are present and up‑to‑date with the implemented code.
- **Repository**: Public, with secret scanning enabled; no credentials or sensitive data are committed.
- **Next Steps**: Add a frontend, implement production authentication, and consider multi‑currency extensions.

---

*This README is intended to serve as the central entry point for recruiters, contributors, and users. It references the existing documentation files for deeper details and does not modify any source code or project history.*
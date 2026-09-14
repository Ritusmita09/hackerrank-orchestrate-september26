# ROADMAP.md — AI Financial Decision Agent

**Version:** 0.1 (planning) · **Date:** 2026-09-14

This roadmap sequences the build from a local prototype to a deployed, multi-user product. The focus is on verifying the deterministic core first, then layering the LLM reasoning on top.

---

## Phase 1: The Deterministic Engine (Weeks 1-2)

**Goal:** A pure, testable financial engine with zero LLM or network dependencies. This is the foundation; if this is broken, the product is unsafe.

- [x] Define core types (Money, Direction, EventStatus, Flexibility, Assumptions).
- [x] Implement `FinancialEvent` normalization and the append-mostly ledger model (state builder).
- [x] Implement first-class amendment records and conflict resolution logging.
- [x] Implement robust recurrence detection (stream-key grouping, month-boundary clamping, conservative estimation).
- [x] Implement the `forecast_balance` pure function.
- [x] Implement safe-amount search, candidate generation, spending-change search, and ranking.
- [x] Implement the `verify_plan` verification gate.
- [x] **Tests:** Unit tests + property-based tests for correct arithmetic, date handling, conflict resolution, deduplication, and verification invariants. (`python run_tests.py` — 30 tests, zero non-stdlib dependencies.)

## Phase 2: Backend + PostgreSQL + CSV/Documents Ingestion (Weeks 3-4)

**Goal:** The API, database, and structured ingestion paths (ready for the AI layer).

- [ ] SQLAlchemy models, Alembic migrations for users, accounts, ledger, patterns, plans, verifications, decisions.
- [ ] FastAPI setup with pydantic-settings config and structlog.
- [ ] API routes for manual data entry, event listing, state summary.
- [ ] Deterministic CSV import pipeline (mapping, deduping, dry-run preview, commit).
- [ ] The LLM provider protocol (`llm/base.py`) + Anthropic adapter.
- [ ] Document extraction pipeline: image/PDF → multimodal LLM → JSON draft → schema validation.
- [ ] Message interpretation pipeline: text → LLM → amendment records → schema validation.
- [ ] Docker compose (API + PostgreSQL).

## Phase 3: LLM Agent + Tool Calling (Weeks 5-6)

**Goal:** A working agent that understands questions, gathers evidence, and produces verified answers.

- [ ] Tool registry and tool-call wrapper.
- [ ] Wire the tools to the engine, db, and ingest pipes.
- [ ] Orchestrator loop (budgeting, routing, response parsing, recovery).
- [ ] Agent prompts and context assembly (state summary, memory).
- [ ] Canned-transcript agent testing (test dispatch and reasoning without burning tokens).

## Phase 4: Memory + Multimodal Understanding (Week 7)

**Goal:** The system remembers preferences, goals, and facts; documents/images are first-class inputs.

- [ ] Memory stores (preferences, goals, facts).
- [ ] Multimodal document understanding refined (payslips, bills, invoices, receipts).
- [ ] User confirmation flows for inferred facts (patterns, extractions, interpretations).

## Phase 5: Streamlit UI (Week 8)

**Goal:** A working interactive frontend.

- [ ] Chat interface with streaming responses.
- [ ] Data tables, CSV/document import wizards, plan review views.
- [ ] Decision audit view (fact sheet, verification evidence, forecast chart).

## Phase 6: Testing + Deployment (Weeks 9-10)

**Goal:** Production-ready, observable, deployed.

- [ ] End-to-end integration tests (mocking the LLM provider for CI).
- [ ] Cost/token accounting per user and per conversation.
- [ ] Trace logging of the agent loop.
- [ ] Complete `docker-compose.yml` for seamless `up` (API, DB, UI).
- [ ] CI/CD pipeline (linting, tests, docker build, deploy).
- [ ] Deployment to managed container service.

## Phase 6 (Future): Integrations & Automation (Post-MVP)

- Direct bank aggregation (Open Banking / Plaid / Salt Edge).
- Automated ingestion streams (IMAP/Gmail connector for bills/receipts, SMS forwarding).
- Periodic push notifications / proactive affordablitiy updates.
- Multi-currency support across engine, models, and UI.
- Native mobile app (replacing the Streamlit frontend with a SPA/native client against the established API).

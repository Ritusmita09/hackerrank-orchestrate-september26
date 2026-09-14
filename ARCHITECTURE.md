# ARCHITECTURE.md — AI Financial Decision Agent

**Version:** 0.1 (planning) · **Date:** 2026-09-14 · **Status:** Pre-implementation

---

## 1. System Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│  Streamlit UI (MVP frontend)                                           │
│  chat · data entry · CSV import · document upload · plan review        │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │ REST / JSON
┌──────────────────────────────▼─────────────────────────────────────────┐
│  FastAPI Backend                                                       │
│                                                                        │
│  ┌──────────────────┐   ┌───────────────────────────────────────────┐ │
│  │  API layer        │   │  Agent Orchestrator (the "brain")         │ │
│  │  /chat /ingest    │──▶│  LLM reasoning loop + tool dispatch       │ │
│  │  /state /plans    │   │  (provider-swappable, see §6)             │ │
│  └──────────────────┘   └───────┬───────────────────────────────────┘ │
│                                  │ typed tool calls                    │
│  ┌───────────────────────────────▼───────────────────────────────────┐ │
│  │  Tool Layer (registered functions, schema-validated args/results) │ │
│  │  state tools · transaction tools · document/image tools ·          │ │
│  │  message tools · simulation tools · verification tools · memory    │ │
│  └───────┬───────────────────────────────┬───────────────────────────┘ │
│          │                               │                             │
│  ┌───────▼────────────┐        ┌─────────▼─────────────────────────┐   │
│  │  Deterministic     │        │  Persistence (SQLAlchemy)         │   │
│  │  Financial Engine  │        │  PostgreSQL                       │   │
│  │  (pure, no LLM,    │        │                                   │   │
│  │   versioned)       │        │                                   │   │
│  └────────────────────┘        └───────────────────────────────────┘   │
│                                                                        │
│  Cross-cutting: auth · observability (traces/tool calls/LLM usage) ·  │
│  config (pydantic-settings) · error handling                           │
└────────────────────────────────────────────────────────────────────────┘
```

**The single most important boundary:** the Deterministic Financial Engine (left) has **zero** LLM dependencies. The agent may *call* it, feed it data, and narrate its results — but never substitute its own arithmetic for the engine's.

## 2. Repository / Service Layout

Monorepo, single deployable backend, designed to split later if needed.

```
ai-financial-agent/
├── PROJECT_SPEC.md / ARCHITECTURE.md / ROADMAP.md / SECURITY_AND_SAFETY.md
├── docker-compose.yml            # app + postgres (+ streamlit)
├── .env.example                  # documented env vars (no secrets)
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI app factory
│   │   ├── config.py             # pydantic-settings; all knobs
│   │   ├── api/                  # routers: chat, ingest, state, plans, documents, health
│   │   ├── agent/
│   │   │   ├── orchestrator.py   # the reasoning loop
│   │   │   ├── prompts.py        # system prompt, per-tool guidance
│   │   │   ├── context.py        # what the LLM sees per turn (state summary, memory)
│   │   │   └── budget.py         # step/token limits per turn
│   │   ├── tools/
│   │   │   ├── registry.py       # tool registration, JSON-schema validation
│   │   │   ├── state_tools.py
│   │   │   ├── transaction_tools.py
│   │   │   ├── document_tools.py
│   │   │   ├── message_tools.py
│   │   │   ├── simulation_tools.py
│   │   │   ├── verification_tools.py
│   │   │   └── memory_tools.py
│   │   ├── engine/               # ★ DETERMINISTIC — no LLM, no network
│   │   │   ├── models.py         # Money, FinancialEvent, RecurringPattern, Plan, ...
│   │   │   ├── state.py          # state reconstruction & conflict resolution
│   │   │   ├── recurrence.py     # pattern detection & occurrence generation
│   │   │   ├── forecast.py       # daily balance forecast (pure function)
│   │   │   ├── affordability.py  # safe amounts, earliest safe dates
│   │   │   ├── planner.py        # candidate strategy generation
│   │   │   ├── verify.py         # ★ the verification gate
│   │   │   ├── ranking.py        # deterministic plan ranking
│   │   │   └── version.py        # ENGINE_VERSION, input hashing
│   │   ├── ingest/
│   │   │   ├── csv_import.py     # mapping, preview, dedup
│   │   │   ├── extraction.py     # doc/image → structured JSON (LLM) → validation
│   │   │   ├── amendments.py     # message → amendment records (LLM) → applier
│   │   │   └── pipeline.py       # orchestration of the above
│   │   ├── llm/                  # provider abstraction (§6)
│   │   │   ├── base.py           # protocol: complete(), complete_structured()
│   │   │   ├── anthropic_client.py
│   │   │   └── factory.py        # chooses provider/model from config
│   │   ├── db/
│   │   │   ├── models.py         # SQLAlchemy ORM
│   │   │   ├── repositories.py   # data access, one per aggregate
│   │   │   └── migrations/       # alembic
│   │   ├── memory/
│   │   │   ├── preferences.py / facts.py / goals.py / conversation.py
│   │   ├── observability/
│   │   │   ├── logging.py        # structlog config
│   │   │   ├── tracing.py        # agent traces, tool calls
│   │   │   └── usage.py          # token/cost accounting
│   │   └── security/             # auth, injection defenses, input guards
│   └── tests/
│       ├── unit/ (engine, ingest parsers)
│       ├── property/ (forecast, verify invariants)
│       ├── agent/ (canned-transcript tool tests)
│       ├── adversarial/ (injection, malformed data)
│       └── safety/ (verification-gate enforcement)
├── frontend/                     # Streamlit app (MVP)
└── docs/
```

## 3. Deterministic Financial Engine (the heart)

### 3.1 Design rules

1. **Pure functions where possible.** `forecast(state, assumptions, horizon) -> CashflowSeries` takes immutable inputs and returns a value object. Same inputs → same outputs, forever.
2. **Money as integers.** `Money` wraps integer minor units + ISO currency code. No floats anywhere in arithmetic. (The old project used floats throughout — a real defect we do not repeat.)
3. **Versioned and hashed.** Every engine result carries `ENGINE_VERSION` and a hash of its inputs, persisted with each decision. A recommendation can be replayed and audited later.
4. **Conservative defaults, explicit assumptions.** Assumptions (income haircut, expense estimator, horizon) are a named, versioned object — never ambient globals.
5. **No I/O.** No DB, no network, no clock reads inside the engine. Time is an input.

### 3.2 Core types (engine/models.py)

```python
class Money:            # integer minor units + currency; Decimal at the edges
class Direction(Enum):  # credit | debit
class EventStatus(Enum):# settled | pending | scheduled | forecasted | cancelled | failed | unrealized
class Flexibility(Enum):# fixed | reducible(min_amount) | stoppable
class Provenance:       # source_type: manual|csv|document|message|pattern, source_id, extracted_by, confidence, confirmed_by_user

class FinancialEvent:   # id, date, direction, Money, category, description,
                        # status, flexibility, provenance, linked_event_id
class RecurringPattern: # stream_key, period (fixed-N-days | monthly(days_of_month) | semimonthly(d1,d2)),
                        # typical_amount + estimator used, anchor, evidence event_ids,
                        # confidence, user_confirmed
class Assumptions:      # horizon_days, income_estimator, expense_estimator,
                        # count_pending_credits (default False), min_balance_floor_policy
class FinancialState:   # user_id, as_of, balances, events, patterns, preferences, goals
class CashflowSeries:   # date -> balance (plus the underlying flows, for explanations)
class Plan:             # method, payments[(date, Money)], total_payable, spending_changes,
                        # option_id, status
class VerificationResult: # plan_id, passed, violations[], evidence[], engine_version, input_hash
```

### 3.3 State reconstruction

- All ingested data lands in the normalized event ledger; `build_state(user_id, as_of)` assembles a `FinancialState` from the ledger:
  - settled events before `as_of` → history;
  - scheduled events at/after `as_of` → confirmed future flows;
  - pending debits → reserved immediately; pending credits → excluded (conservative);
  - cancelled/failed/unrealized → excluded from flows (unrealized investments are not cash).
- **Conflict resolution** is a deterministic, ordered ruleset (explicit amendment > newer record > settled > safer interpretation) and returns a log of which conflicts it resolved and how — that log feeds explanations.
- Amendments (from messages/documents) are stored as **first-class amendment records applied at state-build time**, not by mutating history in place. History is append-mostly; corrections are explicit, reviewable, reversible.

### 3.4 Recurrence detection (reused concept from the old project, hardened)

Kept from the old project because it was sound:
- interval clustering around common periods (7/14/15/28/30/31 days) with tolerance;
- day-of-month and semi-monthly (two anchor days) anchoring with month-length clamping;
- conservative income estimation (low percentile / most-recent with a floor policy) vs median for expenses;
- seeding patterns from scheduled events;
- exclusion of one-off adjustments (bonus/arrears) from stream estimation.

Hardened for production:
- **Grouping by stream, not just category** — the old `(category, direction, flexibility)` key merged distinct salary streams (documented failure in the old repo). New key: normalized description/merchant + category + direction; post-pass splits patterns with multiple distinct anchor days.
- Detection result is a **proposal** with confidence and evidence event ids; it becomes authoritative only after user confirmation (or high-confidence auto-accept with visible flag).
- Every generated future occurrence carries `pattern_id` provenance so a user can correct one stream without touching others.

### 3.5 Forecast

- Daily-granularity balance series over the horizon from: starting balance − pending debits + scheduled events + confirmed/accepted pattern occurrences (deduplicated against scheduled events by stream key + date — the old double-counting bug, fixed structurally by deduplicating on `stream_key`, not `(date, category, direction)`).
- Pure function; series includes the per-day flow breakdown so explanations can say "on Oct 15 your balance dips to X because rent and the card payment both land."

### 3.6 Affordability & planning

- `compute_safe_amount(state, request)`: monotone search over *exact simulations* (verify each candidate amount by full re-forecast — never a closed-form shortcut).
- `find_earliest_safe_date(state, amount)`: first date d where the single payment passes verification.
- Candidate plans generated deterministically: full / partial / each installment option / spending-change-assisted / wait / not-recommended — mirroring the old project's sound "generate candidates, validate each against the modified forecast, rank the survivors" structure.
- Spending-change search: enumerate minimal subsets (size 1→3) of user-authorized flexible changes, preference for fewer changes and less total relief — same minimal-disruption principle as the old project, but restricted to changes the user's preferences explicitly authorize.
- Ranking: fixed lexicographic order (documented, versioned): completes-by-deadline → no spending changes → lower total payable → earlier start → fewer payments → stable tiebreak.

### 3.7 The verification gate (verify.py)

```python
def verify_plan(plan: Plan, state: FinancialState, assumptions: Assumptions) -> VerificationResult
```

- Re-simulates from the canonical state: applies spending changes, applies each payment in order, asserts for **every** day in the horizon: `balance >= user_min_balance`, all protected/essential expenses honored, plan completes by deadline, payments sum/structure valid for the method.
- Returns `violations` (human-readable, machine-structured) and `evidence` (the days, balances, and events that prove safety).
- **The API layer refuses to serialize a recommendation whose stored `VerificationResult.passed != True`.** This is enforced in code (`plans.py` serializer), not in prompt instructions — the LLM literally cannot bypass it.

## 4. Agent Architecture

### 4.1 Orchestrator

A bounded reasoning loop (not a free-running agent):

```
turn(user_message, conversation):
  context = build_context(conversation, memory, state_summary, budget)
  loop (max N steps, max M tokens):
    response = llm.complete(context + tool defs)
    if response has tool_calls:
        results = [dispatch(tool, schema_validated_args) for call]
        append results to context; continue
    else:
        final = response
  post_process(final):
    if final contains a recommendation: must reference a persisted,
      verified plan id — else re-prompt or degrade
  persist trace, usage
```

- **Step/token budget per turn** (config) prevents runaway loops and cost blowups.
- Failure posture: if the loop cannot complete safely, respond "I couldn't complete this analysis" with what was missing — never a guess.
- The orchestrator is deliberately *thin*: routing, dispatch, budgets, persistence. Judgment lives in the prompt + tools, mechanics live in deterministic code.

### 4.2 Tool layer

All tools are registered functions with JSON schemas for arguments **and** results; the registry validates both directions. Tools are grouped by domain, each with its own module (see §2 layout). Tool results are compact, evidence-tagged JSON — small enough for context, rich enough to reason with.

| Group | Tools (MVP) |
|---|---|
| **State** | `get_financial_summary`, `get_balances`, `get_financial_state_detail` (events/patterns with provenance) |
| **Transactions** | `list_transactions` (filter/slice), `search_transactions`, `add_manual_transaction`, `update_transaction`, `delete_transaction` (soft) |
| **Documents/images** | `submit_document` (register upload), `extract_document` (multimodal → structured JSON draft), `confirm_extraction` (user-confirmed → ingest) |
| **Messages** | `interpret_message` (pasted text → amendment records draft), `preview_amendments` (show resulting state diff), `confirm_amendments` |
| **Simulation** | `forecast_balance(horizon)`, `compute_safe_amount(request)`, `find_earliest_safe_date(amount, deadline)`, `list_installment_options(request)` |
| **Verification** | `generate_candidate_plans(request)` (engine), `verify_plan(plan_id)` (engine gate), `rank_plans(plan_ids)` (deterministic) |
| **Memory** | `save_preference`, `get_preferences`, `save_goal`, `list_goals`, `save_fact`, `search_facts`, `get_conversation_summary` |

Write tools (`add_manual_transaction`, `confirm_*`) always record provenance and remain user-visible/editable.

### 4.3 Turn types and model routing

- **Routine turn** (clarification, data lookup, chit-chat): cheap fast model.
- **Reasoning turn** (affordability decision, plan comparison, explanation drafting): capable model.
- **Extraction turn** (documents/images, message interpretation): multimodal capable model, with structured-output enforcement + schema validation + retry-on-invalid.
- Routing is config-driven (`config.py`), per turn type, per provider.

## 5. Data Ingestion Architecture

### 5.1 The universal pipeline

Every ingestion path converges on the same three stages:

```
raw input → [interpretation] → structured draft → [user confirmation] → normalized event ledger
```

- **CSV:** deterministic parser; column mapping UI (Streamlit) → preview → dedup (fuzzy on date+amount+description) → import as events with `provenance=csv:<import_id>`. No LLM required for parsing; optional LLM assist for category suggestion only.
- **Documents/images:** multimodal LLM extracts a **schema-constrained JSON draft** (document type, parties, dates, amounts, line items, currency, confidence per field). Validation: required fields, amount sanity bounds, date sanity, currency consistency. Draft is shown to the user for confirmation, then ingested with `provenance=document:<id>`. Extraction results are cached by content hash (reuse of the old project's sound image-cache idea).
- **Pasted messages:** LLM produces amendment records: `{action: amend|cancel|delay|confirm|inform, target_hint, fields, quotes, confidence}` — with **mandatory quotes** of the source text supporting each field (so the user, and later review tooling, can audit the interpretation). A deterministic applier maps amendments onto events; ambiguous targeting → ask the user. Result: visible before/after state diff.

### 5.2 Why confirmation-first matters

Inferred facts are always drafts until confirmed. This gives us: auditability (every fact has a chain of custody), reversibility (delete a document → its events cascade), and trust (the user sees what the system believes and why).

## 6. AI Model Layer (provider-swappable)

```python
class LLMClient(Protocol):
    def complete(self, messages, *, tools=None, system=None,
                 max_tokens=None, temperature=None) -> LLMResponse: ...
    def complete_structured(self, messages, schema: JSONSchema) -> dict: ...
    # multimodal: messages may carry image blocks
```

- `anthropic_client.py` is the first implementation (tool use, vision, structured output).
- `factory.py` builds clients from config: `LLM_PROVIDER`, `LLM_MODEL_ROUTING = {"chat": "...", "reasoning": "...", "extraction": "..."}`.
- Adapters normalize: tool-call format, usage/token accounting, retry/timeout semantics, and error taxonomy (retryable vs fatal).
- **Nothing outside `llm/` imports a provider SDK.** The agent, tools, and ingestion code speak only to the protocol. Swapping providers = implementing one adapter + config change.
- All calls flow through `observability/usage.py`: tokens in/out, cost estimate per model, per conversation and per user.

## 7. Database Model (PostgreSQL)

| Table | Purpose / key fields |
|---|---|
| `users` | id, auth identity, home_currency, created_at |
| `accounts` | id, user_id, name, type (checking/savings/cash/credit), manual or future `integration_ref` |
| `balance_snapshots` | account_id, as_of, Money, provenance |
| `transactions` | id, user_id, account_id, date, direction, amount_minor, currency, category, description, status, flexibility, min_allowed_amount, linked_event_id, dedup_key |
| `recurring_patterns` | id, user_id, stream_key, period spec, typical amount + estimator, anchor days, evidence (txn ids), confidence, user_confirmed, active |
| `documents` | id, user_id, kind, content_hash, storage ref, mime |
| `extractions` | id, document_id, draft JSON, validation result, confirmed_at, extracted_by (model) |
| `amendments` | id, user_id, source message text, action, target transaction/pattern, fields, quotes, confidence, status (draft/applied/rejected) |
| `requests` | id, user_id, question text, type, amount, deadline, allows_partial, created_at |
| `payment_options` | id, request_id, method, first_payment_date, freq_days, n_payments, payment amount, total_payable, fees |
| `forecast_runs` | id, user_id, request_id?, assumptions JSON, engine_version, input_hash, series JSON |
| `plans` | id, request_id, forecast_run_id, method, payments JSON, total_payable, spending_changes, option_id, rank |
| `verifications` | plan_id, passed, violations JSON, evidence JSON, engine_version, input_hash, verified_at |
| `decisions` | id, request_id, selected plan_id, verification_id, fact_sheet JSON, explanation, created_at — **the audit record** |
| `preferences` | user_id, key, value, protected/reducible/stoppable category sets, payment-method willingness, min balance policy |
| `goals` | id, user_id, kind (save/payoff/purchase), target amount, target date, priority |
| `facts` | id, user_id, text, source, valid_from/valid_to (e.g., "contract ends March 2027") |
| `conversations` / `messages` | chat history per user; role, content, tool_call summaries, usage |
| `agent_traces` | id, conversation_id, turn index, steps JSON (model, prompt hash, tool calls, durations, errors) |
| `llm_calls` | id, trace_id, provider, model, input/output tokens, cost estimate, latency, status |

Notes:

- **`decisions` + `verifications` + `forecast_runs` are append-only** (no updates) — the audit trail.
- Money columns: `amount_minor BIGINT` + `currency CHAR(3)`; never floats.
- Multi-currency: MVP stores the user's home currency; the schema carries currency per row and an `fx_rates` table is stubbed for later (the old project's dated-FX-lookup logic is reusable when multi-currency arrives).
- Alembic migrations from day one; no `create_all` in production paths.

## 8. API Design (FastAPI)

REST + JSON; OpenAPI auto-documented. (MVP frontend is server-rendered-ish Streamlit calling these APIs; a future SPA swaps in without backend change.)

```
POST /api/chat                      → start/continue agent conversation (SSE streaming)
GET  /api/state/summary             → balances, upcoming, patterns
GET  /api/transactions              → filter/slice
POST /api/transactions              → manual add
POST /api/imports/csv               → upload + mapping
POST /api/imports/csv/preview       → parsed preview + dedup report
POST /api/imports/csv/commit
POST /api/documents                 → upload
POST /api/documents/{id}/extract    → extraction draft
POST /api/documents/{id}/confirm
POST /api/messages/interpret        → amendment draft
POST /api/messages/confirm
POST /api/requests                  → affordability request
POST /api/requests/{id}/analyze     → run engine pipeline (candidate plans)
GET  /api/requests/{id}/plans       → verified, ranked plans + explanations
GET  /api/decisions/{id}/audit      → full evidence bundle
GET  /api/goals | /api/preferences  → memory management
GET  /api/health | /api/version
```

Rules: every mutation returns the resulting state diff; every recommendation endpoint returns only verified plans; errors are structured (`{code, message, details}`); request IDs propagate into logs and traces.

## 9. Frontend Architecture (Streamlit, MVP)

- `frontend/` is a thin client over the backend API — **no business logic in the frontend** (it must be replaceable by a SPA later).
- Pages: **Chat** (primary), **Data** (balances/transactions/manual entry), **Imports** (CSV wizard: map → preview → commit), **Documents** (upload → extraction review → confirm), **Decisions** (request history, plan details with evidence expansion), **Settings** (preferences, goals).
- The decision view renders the verification evidence (forecast chart with min-balance floor, payment markers, violation list if any) — transparency is a feature.
- Auth: MVP uses simple session-based login (single-user local mode supported); real authN arrives with deployment (see SECURITY_AND_SAFETY.md).

## 10. Memory Architecture

Four stores, one interface:

| Store | Content | Lifetime | Mechanism |
|---|---|---|---|
| Preferences | protected/reducible/stoppable categories, payment-method willingness, min-balance policy, risk tolerance, currency | durable | `preferences` table; injected into agent context every turn |
| Recurring knowledge | confirmed recurring streams + corrections ("rent moved to the 3rd") | durable | `recurring_patterns` + `facts`; drives forecasting |
| Goals & facts | savings targets, debt payoff plans, dated facts ("contract ends 2027-03") | durable | `goals` / `facts` tables; relevant facts retrieved by date/context |
| Conversation | short-term turns verbatim; long-term summarized | session→durable | last N turns + rolling LLM summary stored on `conversations` |

Memory tools let the agent read/write these explicitly (no silent persistence of user statements except via `save_*` tools, which are surfaced in the UI). Facts carry validity windows so expired facts stop influencing forecasts automatically.

## 11. Observability

- **Logging:** structlog, JSON to stdout; every log line carries `request_id`, `user_id`, `trace_id`, `conversation_id`. No PII/amounts at INFO; amounts only in DEBUG.
- **Tracing:** every agent turn writes an `agent_traces` row with the full step sequence (model, tool calls, args hash, result size, duration, errors) — this is the debugging surface for "why did it say that?"
- **Decisions:** the `decisions` table is the product-level audit trail (inputs hash, engine version, verification, fact sheet, explanation).
- **Usage/cost:** `llm_calls` per call; rollups per conversation/user/day; cost estimates per model; surfaced in a simple admin view. (Directly generalizes the old project's UsageLogger into a persistent, per-user system.)
- **Metrics:** RED metrics per endpoint; tool-call error rates; LLM retry/fallback rates; verification-failure rate (a key quality signal — if candidate plans frequently fail verification, the planner or the data is drifting).
- **Errors:** structured exception taxonomy; engine errors never surface raw to users — mapped to safe messages with trace IDs.

## 12. Configuration & Environments

- `pydantic-settings`: `APP_ENV` (local/staging/prod), DB URL, LLM provider + model routing, horizons, budgets, cost caps, log level — all from env vars; `.env.example` documents every knob.
- Secrets only via environment / secret manager. `.gitignore` covers `.env`, caches, uploads.
- One `docker-compose.yml`: `api`, `worker` (future), `db`, `frontend`; healthchecks on `/api/health`; volumes for uploads (object storage in prod).

## 13. What We Reuse from the Old Project — and What We Reject

**Reuse (conceptually, rewritten):**
- Deterministic safety check over a forecast horizon; payment simulation as the only source of truth.
- Binary search for safe amounts over exact verification.
- Candidate-plan generation + per-plan validation against the modified forecast + lexicographic ranking.
- Minimal-disruption spending-change subset search (bounded at 3 changes).
- Recurrence detection: period clustering, day-of-month/semi-monthly anchoring with clamping, conservative income estimation, scheduled-event seeding, exclusion of one-off adjustments.
- Dated FX lookup (for future multi-currency).
- Extraction caching by content; usage/cost accounting.

**Reject:**
- Keyword-matching message interpretation (`if 'contract' in msg and 'ended' in msg`) — replaced by LLM → structured amendments → deterministic applier. This was the old project's most brittle, dataset-fitting component.
- Batch CSV-in/CSV-out pipeline, in-memory pandas lookups — replaced by a DB and event ledger.
- Float money arithmetic — replaced by integer minor units.
- Pattern grouping that merges income streams — replaced by stream-key grouping + split post-pass.
- Deducation on `(date, category, direction)` — replaced by stream-key dedup (fixes the salary double-count).
- Template-only explanations — replaced by evidence-constrained LLM explanations with a stored fact sheet.
- Grader-driven parameter tuning (the ~120 debug/grid-search scripts) — replaced by property tests and explicit, versioned assumption objects.
- Single-file error fallback that fabricates a "safe" answer on exceptions — the new system degrades loudly, never quietly.

## 14. Key Architectural Decisions (ADR Summary)

| # | Decision | Rationale |
|---|---|---|
| 1 | Verification gate enforced in the serializer, not prompts | Prompts are suggestions; code is a guarantee |
| 2 | Engine is pure & LLM-free | Testability, auditability, reproducibility, trust |
| 3 | Money = integer minor units | Floats caused real precision defects in the old project |
| 4 | Append-only decisions/verifications/forecast runs | Auditable product; regulatory posture from day one |
| 5 | Confirmation-first ingestion | Trust + reversibility + provenance chain |
| 6 | LLM behind internal protocol | Provider swap = config change; also enables canned-transcript testing |
| 7 | Streamlit calls the API only | Frontend is disposable; backend is the product |
| 8 | Amendments as first-class records | Replaces brittle in-place mutation; reviewable, reversible |
| 9 | Monorepo, single service (MVP) | Ship fast; boundaries (engine/agent/llm) allow extraction later |
| 10 | Bounded agent loop with budgets | Cost control and predictable termination |

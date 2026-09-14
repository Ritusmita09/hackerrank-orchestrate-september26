# PROJECT_SPEC.md — AI Financial Decision Agent

**Version:** 0.1 (planning) · **Date:** 2026-09-14 · **Status:** Pre-implementation

---

## 1. Mission

Build a real, deployable AI-powered personal financial agent that:

1. Understands a user's financial question in natural language.
2. Gathers relevant financial evidence from the user's own data (manual entry, CSV bank exports, documents, images, pasted messages).
3. Interprets structured and unstructured information into a normalized, provenance-tracked financial state.
4. Forecasts the user's cash flow over a forecast horizon.
5. Generates candidate payment/spending strategies.
6. **Verifies every candidate with deterministic code** (balance never falls below the user's protected minimum, essential expenses covered, plan completes by deadline).
7. Ranks only the verified-safe options.
8. Explains the recommendation with **evidence-backed reasoning** — every number traceable to a source.

This is a product for real users making real money decisions — not a competition submission.

## 2. Core Principle (Non-Negotiable)

> **LLM/AI = reasoning, interpretation, planning, tool selection, multimodal understanding, explanation.**
> **Deterministic code = financial arithmetic, forecasting, constraint checking, payment simulation, safety verification, final validation.**

Consequences:

- The LLM **never** declares a financial plan safe. Only `verify_plan()` (deterministic) can.
- The LLM **never** produces a number that reaches the user as a *computed financial fact* without the engine recomputing it. LLM outputs that carry financial values (e.g., an amount read off a payslip) are **data to be ingested**, labeled with provenance and confidence, not conclusions.
- Every displayed recommendation links to the forecast run, inputs, engine version, and verification result that produced it (audit trail).

## 3. Target Users

- Individuals who manage money across accounts without dedicated financial tooling.
- People facing a purchase/expense decision ("Can I afford this?", "When can I?") who want an answer that respects their commitments.
- Initially: technically-comfortable early adopters comfortable uploading CSV exports and documents. Later: broader users via bank aggregation.

## 4. MVP Scope

### In scope (MVP)

| Capability | Detail |
|---|---|
| Chat interface | Streamlit app; natural-language financial questions |
| Manual data entry | Balances, income, expenses, debts, commitments |
| CSV import | Bank-export CSVs with column mapping + preview + confirm |
| Document/image upload | Payslips, bills, invoices, receipts → structured extraction (multimodal LLM) → user confirmation |
| Message/text ingestion | Pasted emails/SMS/notifications → structured amendments to the financial state (amount changes, cancellations, delays, confirmations) |
| Financial state reconstruction | Normalized event model, currency handling (single home currency per user in MVP), deduplication |
| Recurrence detection | Deterministic pattern detection over transaction history, user-confirmable |
| Cash-flow forecasting | Daily balance forecast over a configurable horizon (default 90 days) |
| Affordability decisions | Safe-to-pay amounts, earliest safe full-payment date, candidate strategies |
| Payment strategy generation | Full payment, partial payment, installments (user-supplied options), wait, not-recommended |
| Spending-change search | Optional reduction/stop of flexible recurring expenses, minimal-disruption preference |
| Deterministic verification | Every candidate plan simulated; only verified plans shown |
| Explanations | LLM-written, but constrained to engine-provided facts and evidence citations |
| Memory | Preferences, recurring items, goals, important facts, conversation context |
| Audit trail | Every recommendation persisted with inputs, engine version, verification result |
| Observability | Structured logs, agent traces, tool-call logs, LLM token/cost accounting |

### Out of scope (MVP, by design)

- **Direct bank integration** (Open Banking / Plaid / Salt Edge) — architecture must allow it later; not built now.
- **Automated email/SMS ingestion** (Gmail/IMAP, SMS forwarding) — MVP accepts pasted text; later stages add connectors.
- Live market data, investment price prediction, securities recommendations.
- Tax advice, regulated financial advice, credit products, lending.
- Multi-user households with shared ledger logic (single-user accounts only, though the model doesn't preclude sharing).
- Mobile apps, real-time notifications.

## 5. Product Requirements (Hard Constraints)

1. **No hardcoded sample answers.** No response may be pre-baked or dataset-specific.
2. **No competition-specific assumptions.** No grader-shaped heuristics (e.g., keyword matching tuned to specific message phrasings).
3. **No fake data.** The system never invents income, expenses, or options. If data is missing, it says so and asks.
4. **No unsupported claims.** Explanations cite only facts the engine computed or data the user supplied.
5. **Every recommendation verified by deterministic code before display.** No exceptions, no bypass flag.
6. **Evidence-backed explanations.** Each recommendation references the specific balances, events, patterns, and forecast points that justify it.
7. **User confirmation for inferred facts.** Recurring patterns, document extractions, and message interpretations are surfaced to the user for confirmation before they become authoritative inputs.
8. **Conservative-by-default forecasting.** Pending credits are not counted as available; forecast income uses conservative estimates; the user's protected minimum balance is a hard floor in all simulations.
9. **Provider-swappable AI layer.** The LLM backend is behind an internal interface; swapping providers is a config change, not a rewrite.
10. **Secrets via environment variables.** No credentials in code or repo. Ever.

## 6. Functional Requirements

### FR-1 — Question Understanding
- The agent classifies the user's question (affordability, forecast, state inquiry, data update, general help) via LLM reasoning and selects appropriate tools.
- Ambiguous questions trigger clarifying questions, not guesses.

### FR-2 — Evidence Gathering
- Tools expose: current balances, transaction history, recurring patterns, commitments, documents, goals, preferences.
- The agent decides which evidence a question needs; it may ask the user for missing data.

### FR-3 — Unstructured Input Interpretation
- Documents/images → multimodal LLM → **structured JSON** (schema-validated) → user confirms → ingested as financial events with provenance `document:<id>`.
- Pasted messages → LLM → structured amendment records (schema: `{action: amend|cancel|delay|confirm|inform, target, fields, confidence}`) → deterministic applier mutates state → user sees the change.
- All unstructured content is **untrusted** (see SECURITY_AND_SAFETY.md §5).

### FR-4 — Financial State Reconstruction
- All sources merge into a normalized event ledger per user: date, direction, amount (Decimal, minor units internally), category, status (settled/pending/scheduled/forecasted/cancelled/failed), source, confidence.
- Conflict resolution is explicit and explainable: explicit amendment > newer record > settled over estimate > financially safer interpretation.

### FR-5 — Forecasting
- Daily balance forecast over a horizon (default 90 days) from: current balance, scheduled events, confirmed recurring patterns, pending debits. Pending credits excluded.
- Forecast is a **pure function**: `(state, assumptions, engine_version) → cashflow series`. Fully unit-testable, reproducible, persistable.

### FR-6 — Affordability & Strategy
- `compute_safe_amount`: maximum payable today such that all safety constraints hold (binary/monotone search over exact simulation, not heuristics).
- `find_earliest_safe_date`: first date the full amount passes the safety check.
- Candidate generation: full payment, partial payment (if permitted), each user-supplied installment option, spending-change-assisted payment, wait.
- Deterministic ranking with an explicit, documented preference order (completes by deadline > no spending changes > lower total cost > earlier start > fewer payments).

### FR-7 — Verification Gate
- `verify_plan(plan, state) → {pass, violations[], evidence[]}` — re-simulates the plan from scratch against the canonical state. A plan is displayable **only if** `pass == true`.
- The verification result (including engine version and input hash) is persisted with the decision.

### FR-8 — Explanation
- The LLM writes the explanation from **only** the engine-supplied fact sheet (status, plan, verification evidence, key balances/flows). The fact sheet is stored alongside the explanation, so any claim in prose is checkable.

### FR-9 — Memory
- Preferences (protected categories, reducible/stopppable categories, payment-method willingness, min balance, risk tolerance).
- Recurring income/expense patterns (deterministic detection + user confirmation).
- Goals (savings targets, debt payoff, deadlines).
- Important financial facts (employment status changes, upcoming known events).
- Conversation context (recent turns, summarized long-term thread).

### FR-10 — Data Management
- Users can view, edit, and delete every ingested fact and see its provenance.
- Deleting a source (e.g., a document) removes/cascades its derived events.

## 7. Non-Functional Requirements

| Area | Requirement |
|---|---|
| Correctness | Financial engine 100% deterministic; property-based + golden tests; money in integer minor units |
| Latency | Simple chat responses < 5s to first token (streaming); full decision pipeline < 30s p95 |
| Reliability | LLM calls have retry/timeout/fallback; agent degrades to "I can't complete this safely" rather than guessing |
| Security | Secrets in env vars; encryption at rest for DB; TLS in transit; per-user data isolation; injection-hardened ingestion |
| Observability | Structured JSON logs with request/trace IDs; tool-call and LLM-call records; per-user token/cost accounting |
| Portability | Dockerized backend + DB; runs locally with docker compose; deploys to any container host |
| Configurability | All model choices, horizons, thresholds in config (env/pydantic-settings), never hardcoded |
| Cost | Per-conversation token budgets; cheap model default for routine turns, capable model for reasoning/extraction |
| Testability | Engine has zero LLM dependencies; agent tests use recorded/canned LLM transcripts |

## 8. Success Criteria for MVP

1. A user can go from empty account → uploaded CSV + one document + one pasted message → confirmed financial state → an affordability question → a **verified** recommendation with evidence, end to end.
2. 100% of displayed recommendations pass the deterministic verifier (enforced by architecture, confirmed by tests).
3. The financial engine's core (state reconstruction, forecast, safe amounts, plan verification) is fully covered by unit + property tests with **no network or LLM calls**.
4. The whole stack runs via `docker compose up` and the LLM provider can be switched by changing one env var.
5. Every agent turn is traceable: which tools ran, what the LLM saw, what it cost.

## 9. Explicit Non-Goals (Permanent)

- The system is **not** a licensed financial advisor and will always present itself as an informational/decision-support tool (see SECURITY_AND_SAFETY.md §7).
- The system will not execute payments or move money. It recommends; the user acts.
- The system will not store bank credentials. (Future bank integration uses regulated aggregation APIs with token-based access, never stored passwords.)

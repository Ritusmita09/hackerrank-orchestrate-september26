"""SQLAlchemy ORM models — the persistence layer around the deterministic engine.

Design rules (see ARCHITECTURE.md §7):

* **Money is integer minor units** (``amount_minor BIGINT`` + ``currency
  CHAR(3)``). Never floats, never Decimal columns.
* **Append-only audit tables.** ``decisions``, ``verification_results``,
  ``forecast_runs``, ``audit_events`` are never updated or deleted — they are
  the product's audit trail. ``financial_profiles`` is versioned by revision;
  updates append a new revision instead of mutating the old one.
* **Corrections are records, not mutations.** A transaction that amends another
  points at it via ``amends_event_id``; the engine's conflict resolution
  decides which one wins at state-build time.
* Every decision-relevant row carries ``engine_version`` and/or an
  ``input_hash`` so any past recommendation can be replayed and audited.

The engine never imports this module — persistence is translated into the
engine's typed domain models by the service layer.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from sqlalchemy import (
    BigInteger,
    Boolean,
    CHAR,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from .base import build_engine, build_session_factory  # noqa: F401 (re-export)


def _new_id() -> str:
    return uuid.uuid4().hex


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Users and profiles
# ---------------------------------------------------------------------------


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(255), default="")
    home_currency: Mapped[str] = mapped_column(CHAR(3), default="USD")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    profiles: Mapped[list["FinancialProfile"]] = relationship(
        back_populates="user", order_by="FinancialProfile.revision"
    )


class FinancialProfile(Base):
    """Versioned snapshot of the user's balances and authorizations.

    Updates append a new revision; state reconstruction always uses the latest
    revision. Money is minor units in ``currency`` (the user's home currency).
    """

    __tablename__ = "financial_profiles"
    __table_args__ = (
        Index("ix_financial_profiles_user_revision", "user_id", "revision"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    current_balance_minor: Mapped[int] = mapped_column(BigInteger)
    minimum_balance_minor: Mapped[int] = mapped_column(BigInteger)
    currency: Mapped[str] = mapped_column(CHAR(3))
    #: JSON lists of category names the user explicitly authorized.
    protected_categories: Mapped[list] = mapped_column(JSON, default=list)
    reducible_categories: Mapped[list] = mapped_column(JSON, default=list)
    stoppable_categories: Mapped[list] = mapped_column(JSON, default=list)
    #: JSON list of payment-method names (full_payment, wait, ...).
    payment_methods_considered: Mapped[list] = mapped_column(JSON, default=list)
    max_installment_payments: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    user: Mapped[User] = relationship(back_populates="profiles")


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    #: checking | savings | cash | credit
    account_type: Mapped[str] = mapped_column(String(32), default="checking")
    currency: Mapped[str] = mapped_column(CHAR(3))
    balance_minor: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    #: Reserved for a future bank-integration reference. Never credentials.
    integration_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Transaction(Base):
    """One row in the append-mostly normalized event ledger.

    Corrections arrive as new rows with ``amends_event_id`` pointing at the
    row they supersede — history is never mutated in place. ``dedup_key``
    (a hash over the semantic identity of the row) makes imports idempotent.
    """

    __tablename__ = "transactions"
    __table_args__ = (
        UniqueConstraint("user_id", "dedup_key", name="uq_transactions_user_dedup"),
        Index("ix_transactions_user_date", "user_id", "date"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    account_id: Mapped[str | None] = mapped_column(
        ForeignKey("accounts.id"), nullable=True
    )
    date: Mapped[date] = mapped_column(Date)
    #: credit | debit
    direction: Mapped[str] = mapped_column(String(8))
    amount_minor: Mapped[int] = mapped_column(BigInteger)
    currency: Mapped[str] = mapped_column(CHAR(3))
    category: Mapped[str] = mapped_column(String(64), default="uncategorized")
    description: Mapped[str] = mapped_column(Text, default="")
    #: settled | pending | scheduled | forecasted | cancelled | failed | unrealized
    status: Mapped[str] = mapped_column(String(16), default="settled")
    #: fixed | reducible | stoppable
    flexibility: Mapped[str] = mapped_column(String(16), default="fixed")
    min_allowed_amount_minor: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    #: provenance: manual | csv | document | message | pattern | import
    source_type: Mapped[str] = mapped_column(String(16), default="manual")
    source_id: Mapped[str] = mapped_column(String(255), default="unknown")
    confidence: Mapped[float] = mapped_column(default=1.0)
    confirmed_by_user: Mapped[bool] = mapped_column(Boolean, default=True)
    linked_event_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    amends_event_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    #: SHA-256 over (user, date, direction, amount, currency, description) —
    #: set by the service layer; used to reject re-imported duplicates.
    dedup_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class RecurringPattern(Base):
    """A user-visible recurring stream (rent, salary, subscriptions...).

    Detected patterns start as proposals (``user_confirmed=False``); only the
    user (or an explicit confirmation flow) makes them authoritative.
    """

    __tablename__ = "recurring_patterns"
    __table_args__ = (
        Index("ix_recurring_patterns_user_stream", "user_id", "stream_key"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    stream_key: Mapped[str] = mapped_column(String(255))
    category: Mapped[str] = mapped_column(String(64))
    direction: Mapped[str] = mapped_column(String(8))
    #: fixed (period_days) | monthly (period_days_of_month JSON)
    period_type: Mapped[str] = mapped_column(String(16))
    period_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    period_days_of_month: Mapped[list | None] = mapped_column(JSON, nullable=True)
    typical_amount_minor: Mapped[int] = mapped_column(BigInteger)
    estimator: Mapped[str] = mapped_column(String(16))
    currency: Mapped[str] = mapped_column(CHAR(3))
    last_seen: Mapped[date] = mapped_column(Date)
    evidence_event_ids: Mapped[list] = mapped_column(JSON, default=list)
    confidence: Mapped[float] = mapped_column(default=1.0)
    user_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    description: Mapped[str] = mapped_column(Text, default="")
    flexibility: Mapped[str] = mapped_column(String(16), default="fixed")
    min_allowed_amount_minor: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


# ---------------------------------------------------------------------------
# Requests, options, goals, preferences, facts
# ---------------------------------------------------------------------------


class AffordabilityRequest(Base):
    """The user's question: can I pay X by date D?"""

    __tablename__ = "affordability_requests"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    question_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    requested_amount_minor: Mapped[int] = mapped_column(BigInteger)
    currency: Mapped[str] = mapped_column(CHAR(3))
    request_date: Mapped[date] = mapped_column(Date)
    desired_completion_date: Mapped[date] = mapped_column(Date)
    allows_partial_payment: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class PaymentOption(Base):
    """A user-supplied installment option for a request."""

    __tablename__ = "payment_options"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    request_id: Mapped[str | None] = mapped_column(
        ForeignKey("affordability_requests.id"), nullable=True, index=True
    )
    option_key: Mapped[str] = mapped_column(String(64))
    first_payment_date: Mapped[date] = mapped_column(Date)
    frequency_days: Mapped[int] = mapped_column(Integer)
    number_of_payments: Mapped[int] = mapped_column(Integer)
    payment_amount_minor: Mapped[int] = mapped_column(BigInteger)
    total_payable_minor: Mapped[int] = mapped_column(BigInteger)
    currency: Mapped[str] = mapped_column(CHAR(3))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class FinancialGoal(Base):
    __tablename__ = "financial_goals"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    #: save | payoff | purchase
    kind: Mapped[str] = mapped_column(String(32))
    target_amount_minor: Mapped[int] = mapped_column(BigInteger)
    currency: Mapped[str] = mapped_column(CHAR(3))
    target_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    description: Mapped[str] = mapped_column(Text, default="")
    achieved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class UserPreference(Base):
    """Key/value preferences (risk tolerance, display options, budgets...)."""

    __tablename__ = "user_preferences"
    __table_args__ = (
        UniqueConstraint("user_id", "key", name="uq_user_preferences_user_key"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    key: Mapped[str] = mapped_column(String(128))
    value: Mapped[dict | list | str | int | bool | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class FinancialFact(Base):
    """Dated, free-text facts with validity windows ("contract ends 2027-03")."""

    __tablename__ = "financial_facts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    text: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(64), default="manual")
    valid_from: Mapped[date | None] = mapped_column(Date, nullable=True)
    valid_to: Mapped[date | None] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


# ---------------------------------------------------------------------------
# Documents and messages (untrusted input)
# ---------------------------------------------------------------------------


class Document(Base):
    """Metadata for an uploaded document. Bytes go to storage, not the DB."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    #: payslip | bill | invoice | receipt | statement | other
    kind: Mapped[str] = mapped_column(String(32), default="other")
    filename: Mapped[str] = mapped_column(String(255), default="")
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    mime_type: Mapped[str] = mapped_column(String(128))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    storage_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    extractions: Mapped[list["DocumentExtraction"]] = relationship(back_populates="document")


class DocumentExtraction(Base):
    """One extraction draft for a document — a proposal until confirmed.

    Ingested events derived from a confirmed extraction carry provenance
    ``document:<document_id>`` so deleting the document can cascade.
    """

    __tablename__ = "document_extractions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    #: Schema-validated structured draft (amount, dates, parties, confidence...).
    draft: Mapped[dict] = mapped_column(JSON, default=dict)
    validation_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    #: draft | confirmed | rejected
    status: Mapped[str] = mapped_column(String(16), default="draft")
    #: Model name once LLM extraction exists; manual until then.
    extracted_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    confirmed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    document: Mapped[Document] = relationship(back_populates="extractions")


class Message(Base):
    """A pasted message/SMS/email — untrusted input, stored for interpretation.

    Interpretation into amendment records is a Phase 3 (LLM) capability; Phase 2
    stores the raw text (bounded in length) with its status.
    """

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    raw_text: Mapped[str] = mapped_column(Text)
    #: Structured amendment records, once interpreted (Phase 3).
    interpretation: Mapped[list | None] = mapped_column(JSON, nullable=True)
    #: received | interpreted | applied | rejected
    status: Mapped[str] = mapped_column(String(16), default="received")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


# ---------------------------------------------------------------------------
# Audit trail (append-only)
# ---------------------------------------------------------------------------


class ForecastRun(Base):
    """One persisted forecast — replayable via engine_version + input_hash."""

    __tablename__ = "forecast_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    request_id: Mapped[str | None] = mapped_column(
        ForeignKey("affordability_requests.id"), nullable=True
    )
    assumptions: Mapped[dict] = mapped_column(JSON, default=dict)
    engine_version: Mapped[str] = mapped_column(String(32))
    input_hash: Mapped[str] = mapped_column(String(64))
    #: Compact JSON of the daily series (dates, balances, flows).
    series: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class VerificationResult(Base):
    """The verification gate's verdict on one candidate plan. Append-only."""

    __tablename__ = "verification_results"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    request_id: Mapped[str | None] = mapped_column(
        ForeignKey("affordability_requests.id"), nullable=True
    )
    plan_id: Mapped[str] = mapped_column(String(64), index=True)
    passed: Mapped[bool] = mapped_column(Boolean)
    violations: Mapped[list] = mapped_column(JSON, default=list)
    evidence: Mapped[list] = mapped_column(JSON, default=list)
    engine_version: Mapped[str] = mapped_column(String(32))
    input_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Decision(Base):
    """The audit record: what the system recommended, and on what basis.

    A decision references the selected plan (as JSON, exactly as verified),
    the state it was computed against (input_hash), the engine version, and
    the fact sheet the future LLM explanation layer will be constrained to.
    """

    __tablename__ = "decisions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    request_id: Mapped[str] = mapped_column(
        ForeignKey("affordability_requests.id"), index=True
    )
    state_input_hash: Mapped[str] = mapped_column(String(64))
    engine_version: Mapped[str] = mapped_column(String(32))
    #: affordable_now | affordable_with_plan | affordable_later | not_affordable
    status: Mapped[str] = mapped_column(String(32))
    selected_plan: Mapped[dict] = mapped_column(JSON, default=dict)
    fact_sheet: Mapped[list] = mapped_column(JSON, default=list)
    #: LLM-written in Phase 3; deterministic summary until then.
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: e.g. "deterministic-engine-1.0.0" — makes explicit that no LLM
    #: participated in this decision.
    decision_layer: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AuditEvent(Base):
    """Generic, append-only audit log for every consequential action."""

    __tablename__ = "audit_events"
    __table_args__ = (
        Index("ix_audit_events_user_created", "user_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    user_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    request_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    #: user_created | transaction_created | csv_imported | document_registered |
    #: message_received | forecast_run | decision_made | verification_recorded ...
    event_type: Mapped[str] = mapped_column(String(64))
    #: Small, non-sensitive JSON payload (ids, hashes, counts — never secrets,
    #: never raw documents).
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

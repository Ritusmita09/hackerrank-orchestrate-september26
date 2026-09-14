"""Extraction pipeline — document image -> structured JSON draft -> validation.

ARCHITECTURE.md §5.1 defines the documents path: a multimodal LLM extracts a
*schema-constrained JSON draft* (document type, parties, dates, amounts, line
items, currency, confidence per field); a deterministic validator checks
required fields, amount sanity, date sanity, and currency; the draft is stored
for the user to confirm, and only a confirmed extraction is ever ingested.

Three boundaries are enforced here rather than requested in a prompt:

* **The model proposes, the validator disposes.** The LLM's output is a draft,
  never an event. Nothing here writes to the ledger; ingestion happens in the
  confirmation service (``confirmation_service.confirm_extraction``) with
  provenance.
* **Structured output is enforced, with one retry.** The schema rides in the
  prompt; a reply that is not valid JSON matching the schema gets exactly one
  corrective round-trip, then a domain error — never a silently mangled draft.
* **Extractions are cached by content hash.** Re-extracting the same bytes
  reuses the stored draft instead of paying for another model call.

Everything is deterministic apart from the injected ``LLMProvider`` call
itself, which is why the whole pipeline is unit-testable with a canned LLM.
"""
from __future__ import annotations

import base64
import json
import re
from datetime import date
from typing import Any, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import Document, DocumentExtraction, User
from .audit_service import record_audit_event
from .document_service import (
    InputServiceError,
    get_document,
    read_document_content,
)


class ExtractionServiceError(Exception):
    """Domain error carrying a machine-readable code (service convention)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# The draft schema — what the model must produce, and what gets stored.
# ---------------------------------------------------------------------------

#: Document kinds the draft may claim (aligned with Document.kind).
DOCUMENT_TYPES = ("payslip", "bill", "invoice", "receipt", "statement", "other")

#: Sanity bounds for any single amount (minor units): positive and below
#: 10 million of a major unit — a document claiming more is a finding.
MAX_AMOUNT_MINOR = 10_000_000_000  # e.g. $10,000,000.00

#: Dates outside this window are findings, not truth.
EARLIEST_PLAUSIBLE_DATE = date(1990, 1, 1)
FUTURE_TOLERANCE_DAYS = 5 * 365

_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")


class LineItem(BaseModel):
    description: str = ""
    amount_minor: int = Field(..., description="Line amount in minor units (integer).")


class ExtractionDraft(BaseModel):
    """The schema-constrained draft a document extraction must produce.

    Amounts are integer minor units, exactly like the engine's Money — the
    model is instructed to emit integers, and fractional values are rejected
    by the schema validator (with one retry) rather than rounded, because
    rounding a model's guess would silently invent money.
    """

    document_type: str
    issuer: str = ""
    issue_date: Optional[date] = None
    due_date: Optional[date] = None
    currency: str
    total_amount_minor: int
    line_items: List[LineItem] = Field(default_factory=list)
    #: Per-field confidence the model reports for its own extraction, 0..1.
    field_confidence: dict[str, float] = Field(default_factory=dict)
    notes: str = ""

    @field_validator("document_type")
    @classmethod
    def _check_document_type(cls, value: str) -> str:
        cleaned = (value or "").strip().lower()
        if cleaned not in DOCUMENT_TYPES:
            raise ValueError(
                f"document_type must be one of: {', '.join(DOCUMENT_TYPES)}"
            )
        return cleaned

    @field_validator("currency")
    @classmethod
    def _check_currency(cls, value: str) -> str:
        cleaned = (value or "").strip().upper()
        if not _CURRENCY_RE.match(cleaned):
            raise ValueError("currency must be a 3-letter ISO code, e.g. 'USD'")
        return cleaned

    @field_validator("total_amount_minor")
    @classmethod
    def _check_total(cls, value: int) -> int:
        if value <= 0 or value > MAX_AMOUNT_MINOR:
            raise ValueError(
                f"total_amount_minor must be between 1 and {MAX_AMOUNT_MINOR}"
            )
        return value

    @field_validator("field_confidence")
    @classmethod
    def _check_confidence(cls, value: dict[str, float]) -> dict[str, float]:
        for key, confidence in value.items():
            if not 0.0 <= float(confidence) <= 1.0:
                raise ValueError(f"confidence for {key!r} must be between 0 and 1")
        return value


def draft_to_json(draft: ExtractionDraft) -> dict[str, Any]:
    """Serialize a draft for storage/tool output (dates become ISO strings)."""
    return draft.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Deterministic validation — runs on every schema-valid draft (ARCHITECTURE §5.1).
# ---------------------------------------------------------------------------

def validate_draft(draft: ExtractionDraft, as_of: date) -> dict[str, Any]:
    """Check amount sanity, date sanity, and line-item consistency.

    Returns ``{"passed": bool, "findings": [...]}``. Findings do not block
    storage — the draft is shown to the user with its findings, and the user
    decides (confirmation flow). Only schema violations block storage.
    """
    findings: List[dict[str, str]] = []

    if draft.total_amount_minor <= 0 or draft.total_amount_minor > MAX_AMOUNT_MINOR:
        findings.append({
            "code": "amount_out_of_bounds",
            "field": "total_amount_minor",
            "message": f"total {draft.total_amount_minor} is outside sane bounds "
                       f"(1..{MAX_AMOUNT_MINOR} minor units)",
        })

    for index, item in enumerate(draft.line_items):
        if item.amount_minor <= 0 or item.amount_minor > MAX_AMOUNT_MINOR:
            findings.append({
                "code": "amount_out_of_bounds",
                "field": f"line_items[{index}].amount_minor",
                "message": f"line item {index} amount {item.amount_minor} is "
                           f"outside sane bounds",
            })

    if draft.issue_date is not None:
        if draft.issue_date < EARLIEST_PLAUSIBLE_DATE:
            findings.append({
                "code": "date_out_of_bounds",
                "field": "issue_date",
                "message": f"issue_date {draft.issue_date} is implausibly old",
            })
        if draft.issue_date > date.fromordinal(
            as_of.toordinal() + FUTURE_TOLERANCE_DAYS
        ):
            findings.append({
                "code": "date_out_of_bounds",
                "field": "issue_date",
                "message": f"issue_date {draft.issue_date} is implausibly far in the future",
            })

    if draft.issue_date is not None and draft.due_date is not None:
        if draft.due_date < draft.issue_date:
            findings.append({
                "code": "date_order_invalid",
                "field": "due_date",
                "message": f"due_date {draft.due_date} is before issue_date "
                           f"{draft.issue_date}",
            })

    if draft.line_items:
        line_sum = sum(item.amount_minor for item in draft.line_items)
        if line_sum != draft.total_amount_minor:
            findings.append({
                "code": "line_items_sum_mismatch",
                "field": "line_items",
                "message": f"line items sum to {line_sum} but the total is "
                           f"{draft.total_amount_minor}",
            })

    return {"passed": not findings, "findings": findings}


# ---------------------------------------------------------------------------
# Structured-output enforcement: prompt, tolerant parse, one retry.
# ---------------------------------------------------------------------------

#: MIME types the multimodal wire format can carry as image blocks.
SUPPORTED_EXTRACTION_MIME_TYPES = ("image/png", "image/jpeg")

_EXTRACTION_PROMPT_TEMPLATE = """You are a document data-extraction engine.
Extract the financial fields from the attached document image.

Return ONLY a single JSON object matching this schema (no prose, no markdown):
{schema}

Rules:
- Amounts are INTEGER minor units (cents): 120.50 USD becomes 12050.
- Dates are ISO YYYY-MM-DD. Use null when a field is not present in the document.
- currency is a 3-letter ISO code.
- document_type is one of: {document_types}.
- field_confidence maps each extracted field name to your confidence from 0.0 to 1.0.
- Report only what the document shows. Never estimate or invent values."""


def build_extraction_prompt() -> str:
    schema = json.dumps(ExtractionDraft.model_json_schema(), indent=2)
    return _EXTRACTION_PROMPT_TEMPLATE.format(
        schema=schema, document_types=", ".join(DOCUMENT_TYPES)
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the JSON object out of a model reply.

    Models wrap JSON in prose or code fences despite instructions, so the
    parse is tolerant: try the whole text, then a fenced block, then the first
    balanced ``{...}`` span. Raises ``ValueError`` when nothing parses — the
    caller turns that into the retry round.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("empty response")

    # Direct parse.
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # ```json ... ``` (or bare ``` ... ```) fenced block.
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        try:
            parsed = json.loads(fence.group(1).strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # First balanced object span.
    start = cleaned.find("{")
    if start >= 0:
        depth = 0
        for index in range(start, len(cleaned)):
            if cleaned[index] == "{":
                depth += 1
            elif cleaned[index] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(cleaned[start:index + 1])
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        break
        raise ValueError("found a '{' but no balanced JSON object followed")
    raise ValueError("no JSON object in the response")


def parse_draft(text: str) -> ExtractionDraft:
    """Parse + schema-validate one model reply. Raises ValueError on failure."""
    payload = extract_json_object(text)
    try:
        return ExtractionDraft.model_validate(payload)
    except ValidationError as exc:
        # Compact, bounded summary of what failed — never the whole payload.
        errors = "; ".join(
            f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}"
            for err in exc.errors()[:5]
        )
        raise ValueError(f"schema violations: {errors}") from None


# ---------------------------------------------------------------------------
# The pipeline: cache check -> read bytes -> model call(s) -> validate -> store.
# ---------------------------------------------------------------------------

def get_cached_extraction(db: Session, user: User, content_hash: str) -> Optional[DocumentExtraction]:
    """The newest reusable extraction for these exact bytes, if any.

    Reuses drafts and confirmations; rejected drafts are re-extracted, because
    a rejection means the user said the result was wrong.
    """
    return db.scalar(
        select(DocumentExtraction)
        .join(Document, Document.id == DocumentExtraction.document_id)
        .where(Document.user_id == user.id)
        .where(Document.content_hash == content_hash)
        .where(DocumentExtraction.status.in_(["draft", "confirmed"]))
        .order_by(DocumentExtraction.created_at.desc(), DocumentExtraction.id.desc())
    )


def _model_name(llm: Any) -> str:
    return str(getattr(llm, "model", "") or "unknown")


def extract_document(
    db: Session,
    user: User,
    *,
    document_id: str,
    llm: Any,
    as_of: date,
    storage_root: Optional[str] = None,
) -> Tuple[DocumentExtraction, bool]:
    """Run multimodal extraction on one uploaded document.

    Returns ``(extraction, cached)``. The stored draft is a proposal: status
    ``draft``, visible to the user, never ingested until confirmed.

    Raises:
        ExtractionServiceError: the document does not exist for this user, the
            MIME type is not extractable, the stored bytes are missing, or the
            model could not produce a schema-valid draft in two attempts.
    """
    try:
        document = get_document(db, user.id, document_id)
    except InputServiceError as exc:
        raise ExtractionServiceError(exc.code, exc.message) from None

    if document.mime_type not in SUPPORTED_EXTRACTION_MIME_TYPES:
        raise ExtractionServiceError(
            "extraction_not_supported",
            f"extraction currently supports images ({', '.join(SUPPORTED_EXTRACTION_MIME_TYPES)}); "
            f"this document is {document.mime_type!r}",
        )

    cached = get_cached_extraction(db, user, document.content_hash)
    if cached is not None:
        return cached, True

    try:
        content = read_document_content(
            user.id, document.content_hash, storage_root=storage_root
        )
    except InputServiceError as exc:
        raise ExtractionServiceError(exc.code, exc.message) from None

    # Imported here, not at module level: app.agent.orchestrator imports the
    # tools package, whose __init__ imports the tool modules, which import
    # this service — a top-level import would close that cycle mid-init.
    from ..agent.orchestrator import Attachment, Message, Role

    attachment = Attachment(
        media_type=document.mime_type,
        data_b64=base64.b64encode(content).decode("ascii"),
    )
    messages: List[Message] = [
        Message(
            role=Role.USER,
            content=build_extraction_prompt(),
            attachments=[attachment],
        )
    ]

    draft = _extract_with_retry(llm, messages)
    validation = validate_draft(draft, as_of)

    extraction = DocumentExtraction(
        document_id=document.id,
        draft=draft_to_json(draft),
        validation_result=validation,
        status="draft",
        extracted_by=_model_name(llm),
    )
    db.add(extraction)
    record_audit_event(
        db,
        event_type="document_extraction_saved",
        user_id=user.id,
        payload={
            "document_id": document.id,
            "extraction_id": extraction.id,
            "validation_passed": validation["passed"],
            "extracted_by": extraction.extracted_by,
        },
    )
    db.flush()
    return extraction, False


def _extract_with_retry(llm: Any, messages: List["Message"]) -> ExtractionDraft:
    """Up to two structured-output attempts; a domain error after that.

    ARCHITECTURE.md §4.3: extraction turns get structured-output enforcement
    with retry-on-invalid. One retry — a model that fails twice will fail
    five times, and every extra round-trip costs money for garbage.
    """
    # Function-local for the same circular-import reason as in extract_document.
    from ..agent.orchestrator import Message, Role

    last_error = "no response"
    for attempt in range(2):
        response = llm.generate(messages)
        raw_text = getattr(response, "text_content", None) or ""
        try:
            return parse_draft(raw_text)
        except ValueError as exc:
            last_error = str(exc)
            if attempt == 0:
                messages = messages + [
                    Message(role=Role.ASSISTANT, content=raw_text),
                    Message(
                        role=Role.USER,
                        content=(
                            "That reply was not valid: "
                            f"{last_error}\n"
                            "Respond again with ONLY the JSON object matching "
                            "the schema. Amounts must be integer minor units."
                        ),
                    ),
                ]

    raise ExtractionServiceError(
        "invalid_draft",
        f"the model could not produce a schema-valid draft in two attempts "
        f"({last_error})",
    )


# ---------------------------------------------------------------------------
# Reads — user-scoped, for the API and the agent tools.
# ---------------------------------------------------------------------------

def list_extractions(
    db: Session, user: User, *, document_id: Optional[str] = None
) -> Sequence[DocumentExtraction]:
    """Extractions for a user's documents (optionally one document), newest first.

    Scoped through the documents join: a user can only ever see extractions of
    documents they own.
    """
    query = (
        select(DocumentExtraction)
        .join(Document, Document.id == DocumentExtraction.document_id)
        .where(Document.user_id == user.id)
        .order_by(DocumentExtraction.created_at.desc(), DocumentExtraction.id.desc())
    )
    if document_id is not None:
        query = query.where(DocumentExtraction.document_id == document_id)
    return db.scalars(query).all()


def get_extraction(db: Session, user: User, extraction_id: str) -> DocumentExtraction:
    """One extraction, scoped to the user's own documents."""
    extraction = db.scalar(
        select(DocumentExtraction)
        .join(Document, Document.id == DocumentExtraction.document_id)
        .where(DocumentExtraction.id == extraction_id)
        .where(Document.user_id == user.id)
    )
    if extraction is None:
        raise ExtractionServiceError(
            "extraction_not_found", f"extraction {extraction_id!r} does not exist"
        )
    return extraction

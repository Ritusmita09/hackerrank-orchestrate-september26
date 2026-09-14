"""Document tools — the agent's interface to document extraction (Phase 4).

ARCHITECTURE.md §4.2 lists the documents tool group: ``submit_document``,
``extract_document`` (implemented in the multimodal segment), and
``confirm_extraction`` (this segment) — the user-confirmed → ingest path.

The extraction tool deliberately builds its *own* provider via the factory
rather than reusing the chat model: extraction turns route to the configured
extraction model and advertise no tools (ARCHITECTURE.md §4.3 turn types).
When no LLM is configured the tool returns a structured failure — it never
fakes an extraction.

``confirm_extraction`` is the only tool that moves a document into the
ledger, and only after the user has seen the draft. It calls the
confirmation service, which enforces the one-ingest-per-draft rule,
provenance, and the home-currency guard; the tool itself adds no policy.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.models import User
from app.agent.tools.base import BaseTool


class ExtractDocumentArgs(BaseModel):
    document_id: str = Field(
        ..., description="The id of the previously uploaded document to extract."
    )


class ExtractDocumentTool(BaseTool):
    name = "extract_document"
    description = (
        "Run multimodal extraction on an uploaded document image (payslip, bill, "
        "invoice, receipt) to produce a structured draft (type, parties, dates, "
        "amounts, line items, currency, confidence). The draft is a proposal with "
        "a validation report - it is never ingested into the ledger until the "
        "user confirms it. Only PNG and JPEG documents are supported."
    )
    args_schema = ExtractDocumentArgs

    def execute(self, db: Session, user: User, args: dict[str, Any], as_of: date) -> Any:
        # Imports here keep the tools package import graph acyclic at load time.
        from app.agent.llm.factory import create_extraction_provider
        from app.agent.llm.omniroute_provider import OmniRouteError
        from app.services.extraction_service import (
            ExtractionServiceError,
            extract_document,
        )

        llm = create_extraction_provider()
        if llm is None:
            return {
                "success": False,
                "error_code": "llm_not_configured",
                "message": "No LLM provider is configured for extraction "
                           "(set LLM_PROVIDER and the OMNIROUTE_* variables).",
            }

        try:
            extraction, cached = extract_document(
                db, user, document_id=args["document_id"], llm=llm, as_of=as_of
            )
        except ExtractionServiceError as exc:
            return {"success": False, "error_code": exc.code, "message": exc.message}
        except OmniRouteError as exc:
            return {"success": False, "error_code": "llm_error", "message": str(exc)}
        finally:
            close = getattr(llm, "close", None)
            if callable(close):
                close()

        return {
            "success": True,
            "extraction_id": extraction.id,
            "document_id": extraction.document_id,
            "cached": cached,
            "status": extraction.status,
            "draft": extraction.draft,
            "validation": extraction.validation_result,
            "extracted_by": extraction.extracted_by,
            "note": "Draft only - present it to the user; nothing is ingested "
                    "until they confirm.",
        }


class ConfirmExtractionArgs(BaseModel):
    extraction_id: str = Field(
        ..., description="The id of the draft extraction the user has approved."
    )
    direction: Optional[str] = Field(
        None,
        description=(
            "User correction for the ledger direction ('credit' or 'debit'). "
            "Only needed when the document type has no default."
        ),
    )
    category: Optional[str] = Field(
        None, description="User correction for the ledger category."
    )
    description: Optional[str] = Field(
        None, description="User correction for the ledger description."
    )
    event_date: Optional[date] = Field(
        None,
        description=(
            "User correction for the event date. Defaults to the draft's due "
            "date (bills/invoices) or issue date (payslips/receipts)."
        ),
    )


class ConfirmExtractionTool(BaseTool):
    name = "confirm_extraction"
    description = (
        "Confirm a document-extraction draft the user has reviewed and approved, "
        "ingesting it into the ledger as one transaction with document provenance. "
        "Call this ONLY after showing the user the draft and getting their "
        "approval - never to tidy up a rejected or already-confirmed extraction. "
        "The user may correct direction, category, description, or the event date; "
        "the amount always comes from the validated draft."
    )
    args_schema = ConfirmExtractionArgs

    def execute(self, db: Session, user: User, args: dict[str, Any], as_of: date) -> Any:
        from app.services.confirmation_service import (
            ConfirmationServiceError,
            confirm_extraction,
        )

        event_date = args.get("event_date")
        if isinstance(event_date, str):
            event_date = date.fromisoformat(event_date)

        try:
            extraction, txn = confirm_extraction(
                db,
                user,
                extraction_id=args["extraction_id"],
                as_of=as_of,
                direction=args.get("direction"),
                category=args.get("category"),
                description=args.get("description"),
                event_date=event_date,
            )
        except ConfirmationServiceError as exc:
            return {"success": False, "error_code": exc.code, "message": exc.message}

        return {
            "success": True,
            "extraction_id": extraction.id,
            "document_id": extraction.document_id,
            "transaction_id": txn.id,
            "date": txn.date.isoformat(),
            "direction": txn.direction,
            "amount_minor": txn.amount_minor,
            "currency": txn.currency,
            "status": txn.status,
            "note": "Confirmed and ingested with document provenance "
                    f"(document:{extraction.document_id}).",
        }

from datetime import date
from typing import Any, Optional

from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from .users import get_db

router = APIRouter()

class DocumentResponse(BaseModel):
    id: str
    filename: str
    kind: str
    status: str

@router.post("", response_model=DocumentResponse)
async def upload_document(
    user_id: str,
    file: UploadFile = File(...),
    kind: str = "other",
    db: Session = Depends(get_db)
):
    """Upload a document (e.g. payslip) for subsequent extraction.

    Bytes are persisted under the storage root; the database keeps metadata
    plus the content hash used for dedup and extraction caching.
    """
    from ...services.user_service import get_user
    from ...services.document_service import register_document

    user = get_user(db, user_id)
    content = await file.read()
    doc = register_document(
        db, user,
        filename=file.filename,
        mime_type=file.content_type or "application/octet-stream",
        content=content,
        kind=kind,
    )
    db.commit()
    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        kind=doc.kind,
        status="registered"
    )


class ExtractionResponse(BaseModel):
    id: str
    document_id: str
    status: str
    cached: bool
    draft: dict
    validation_result: Optional[dict]
    extracted_by: Optional[str]


@router.post("/{document_id}/extract", response_model=ExtractionResponse)
def extract_document_route(
    user_id: str,
    document_id: str,
    db: Session = Depends(get_db)
):
    """Run multimodal extraction on an uploaded document image.

    Returns a schema-validated draft plus a deterministic validation report.
    The draft is a proposal: nothing is ingested until the user confirms it.
    """
    from ...services.user_service import get_user
    from ...services.extraction_service import (
        ExtractionServiceError,
        extract_document,
    )
    from ...agent.llm.factory import create_extraction_provider
    from ...agent.llm.omniroute_provider import OmniRouteError

    user = get_user(db, user_id)

    llm = create_extraction_provider()
    if llm is None:
        raise HTTPException(
            status_code=503,
            detail="No LLM provider is configured for extraction "
                   "(set LLM_PROVIDER and the OMNIROUTE_* variables).",
        )

    try:
        extraction, cached = extract_document(
            db, user, document_id=document_id, llm=llm, as_of=date.today()
        )
        db.commit()
    except ExtractionServiceError as exc:
        db.rollback()
        status = 404 if exc.code in ("document_not_found",) else 400
        raise HTTPException(status_code=status, detail=exc.message) from None
    except OmniRouteError as exc:
        db.rollback()
        raise HTTPException(status_code=502, detail=str(exc)) from None
    finally:
        close = getattr(llm, "close", None)
        if callable(close):
            close()

    return ExtractionResponse(
        id=extraction.id,
        document_id=extraction.document_id,
        status=extraction.status,
        cached=cached,
        draft=extraction.draft,
        validation_result=extraction.validation_result,
        extracted_by=extraction.extracted_by,
    )


@router.get("/{document_id}/extractions", response_model=list[ExtractionResponse])
def list_extractions_route(
    user_id: str,
    document_id: str,
    db: Session = Depends(get_db)
):
    """List the extraction drafts stored for one of the user's documents."""
    from ...services.user_service import get_user
    from ...services.extraction_service import list_extractions

    user = get_user(db, user_id)
    return [
        ExtractionResponse(
            id=extraction.id,
            document_id=extraction.document_id,
            status=extraction.status,
            cached=False,   # listing is a read; cache status applies to a fresh run
            draft=extraction.draft,
            validation_result=extraction.validation_result,
            extracted_by=extraction.extracted_by,
        )
        for extraction in list_extractions(db, user, document_id=document_id)
    ]


class ConfirmExtractionRequest(BaseModel):
    direction: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    event_date: Optional[date] = None


class ConfirmedExtractionResponse(BaseModel):
    extraction_id: str
    document_id: str
    status: str
    transaction_id: str
    date: date
    direction: str
    amount_minor: int
    currency: str
    transaction_status: str


@router.post(
    "/{document_id}/extractions/{extraction_id}/confirm",
    response_model=ConfirmedExtractionResponse,
)
def confirm_extraction_route(
    user_id: str,
    document_id: str,
    extraction_id: str,
    body: Optional[ConfirmExtractionRequest] = None,
    db: Session = Depends(get_db),
):
    """Confirm a draft extraction and ingest it as one ledger transaction.

    The draft's amount and currency are ingested as validated; the caller may
    correct direction, category, description, and the event date. Only
    draft-status extractions can be confirmed.
    """
    from ...services.user_service import get_user
    from ...services.extraction_service import ExtractionServiceError, get_extraction
    from ...services.confirmation_service import (
        ConfirmationServiceError,
        confirm_extraction,
    )

    user = get_user(db, user_id)
    body = body or ConfirmExtractionRequest()

    # The route is document-scoped: an extraction id from another document is
    # a 404, even though the confirmation service would find it by id alone.
    try:
        existing = get_extraction(db, user, extraction_id)
    except ExtractionServiceError as exc:
        raise HTTPException(status_code=404, detail=exc.message) from None
    if existing.document_id != document_id:
        raise HTTPException(
            status_code=404,
            detail=f"extraction {extraction_id!r} does not belong to document "
                   f"{document_id!r}",
        )

    try:
        extraction, txn = confirm_extraction(
            db,
            user,
            extraction_id=extraction_id,
            as_of=date.today(),
            direction=body.direction,
            category=body.category,
            description=body.description,
            event_date=body.event_date,
        )
        db.commit()
    except ConfirmationServiceError as exc:
        db.rollback()
        status = 404 if "not_found" in exc.code else (
            409 if exc.code == "duplicate_event" else 400
        )
        raise HTTPException(status_code=status, detail=exc.message) from None

    return ConfirmedExtractionResponse(
        extraction_id=extraction.id,
        document_id=extraction.document_id,
        status=extraction.status,
        transaction_id=txn.id,
        date=txn.date,
        direction=txn.direction,
        amount_minor=txn.amount_minor,
        currency=txn.currency,
        transaction_status=txn.status,
    )


@router.post("/{document_id}/extractions/{extraction_id}/reject")
def reject_extraction_route(
    user_id: str,
    document_id: str,
    extraction_id: str,
    db: Session = Depends(get_db),
):
    """Reject a draft extraction: the model's read of the document was wrong.

    A rejected draft is re-extracted on the next extraction run (the content-
    hash cache only reuses draft and confirmed rows).
    """
    from ...services.user_service import get_user
    from ...services.extraction_service import ExtractionServiceError, get_extraction
    from ...services.confirmation_service import (
        ConfirmationServiceError,
        reject_extraction,
    )

    user = get_user(db, user_id)

    try:
        existing = get_extraction(db, user, extraction_id)
    except ExtractionServiceError as exc:
        raise HTTPException(status_code=404, detail=exc.message) from None
    if existing.document_id != document_id:
        raise HTTPException(
            status_code=404,
            detail=f"extraction {extraction_id!r} does not belong to document "
                   f"{document_id!r}",
        )

    try:
        extraction = reject_extraction(db, user, extraction_id=extraction_id)
        db.commit()
    except ConfirmationServiceError as exc:
        db.rollback()
        status = 404 if "not_found" in exc.code else 400
        raise HTTPException(status_code=status, detail=exc.message) from None

    return {"extraction_id": extraction.id, "status": extraction.status}

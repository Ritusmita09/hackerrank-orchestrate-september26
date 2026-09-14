from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from .users import get_db
from ..schemas import TransactionCreate, TransactionResponse

router = APIRouter()


@router.post("", response_model=TransactionResponse)
def create_transaction(
    user_id: str,
    req: TransactionCreate,
    db: Session = Depends(get_db)
):
    """Ingest a single manually provided transaction."""
    from ...services.user_service import get_user
    from ...services.transaction_service import ingest_transaction

    user = get_user(db, user_id)
    txn = ingest_transaction(
        db, user,
        date_=req.date,
        direction=req.direction,
        amount_minor=req.amount_minor,
        category=req.category,
        description=req.description,
        status=req.status,
        flexibility=req.flexibility,
        source_type=req.source_type,
        source_id=req.source_id,
    )
    db.commit()
    return txn


@router.get("", response_model=list[TransactionResponse])
def get_transactions(
    user_id: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    from ...services.user_service import get_user
    from ...services.transaction_service import get_transactions as svc_get

    user = get_user(db, user_id)
    txns = svc_get(db, user.id, start_date=start_date, end_date=end_date, status=status)
    return txns


@router.post("/csv")
async def upload_csv_transactions(
    user_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Ingest a batch of transactions securely via CSV."""
    from ...services.user_service import get_user
    from ...services.ingestion_service import import_transactions_csv, CsvImportError

    user = get_user(db, user_id)
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported")

    # Read as text
    try:
        content = (await file.read()).decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Invalid file encoding; must be UTF-8")

    try:
        result = import_transactions_csv(db, user, content, source_id=file.filename)
        db.commit()
    except CsvImportError as exc:
        raise HTTPException(status_code=400, detail={"code": exc.code, "message": exc.message})

    return {
        "status": "success",
        "imported": result.imported,
        "rejected": result.rejected,
        "duplicates": result.duplicates,
        "errors": [{"row": e.row_number, "message": e.message} for e in result.errors]
    }

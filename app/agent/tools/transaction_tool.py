import datetime
from datetime import date
from typing import Any, Optional
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from app.db.models import User
from app.services.transaction_service import ingest_transaction, TransactionServiceError
from app.agent.tools.base import BaseTool

class TransactionToolArgs(BaseModel):
    date: datetime.date = Field(..., description="The date of the transaction.")
    direction: str = Field(..., description="The direction: 'credit' (money in) or 'debit' (money out).")
    amount_minor: int = Field(..., description="The amount in minor units.")
    category: str = Field(..., description="Category of the transaction.")
    description: str = Field("", description="Description of the transaction.")
    status: str = Field("settled", description="Status: 'settled', 'pending', etc.")

class TransactionTool(BaseTool):
    name = "add_transaction"
    description = "Add a manual financial transaction to the ledger (either settled past event or an upcoming pending one)."
    args_schema = TransactionToolArgs

    def execute(self, db: Session, user: User, args: dict[str, Any], as_of: date) -> Any:
        try:
            txn = ingest_transaction(
                db,
                user,
                date_=args["date"],
                direction=args["direction"],
                amount_minor=args["amount_minor"],
                category=args["category"],
                description=args["description"],
                status=args["status"],
            )
            return {
                "success": True,
                "transaction_id": txn.id,
                "message": f"Successfully ingested transaction {txn.id}"
            }
        except TransactionServiceError as e:
            return {
                "success": False,
                "error_code": e.code,
                "message": e.message
            }

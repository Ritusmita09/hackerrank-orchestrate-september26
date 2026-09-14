import datetime
from datetime import date
from typing import Any
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.db.models import User
from app.services.financial_state_service import get_financial_state, state_to_json
from app.agent.tools.base import BaseTool

class StateToolArgs(BaseModel):
    pass

class StateTool(BaseTool):
    name = "get_financial_state"
    description = "Retrieve the current deterministic summary of the user's financial state (balances, history events, future confirmed events, income/expense streams)."
    args_schema = StateToolArgs

    def execute(self, db: Session, user: User, args: dict[str, Any], as_of: date) -> Any:
        state = get_financial_state(db, user, as_of=as_of)
        return state_to_json(state)

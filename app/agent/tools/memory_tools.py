"""Memory tools - the agent's explicit read/write interface to memory.

ARCHITECTURE.md §4.2 lists the memory tool group: "save_preference",
"get_preferences", "save_goal", "list_goals", "save_fact",
"search_facts".

Why tools rather than automatic persistence: memory is written only when a
call is made deliberately, so every change is explicit, visible, and audited
(ARCHITECTURE.md §10 - "no silent persistence of user statements"). An LLM
cannot smuggle a claim into durable state as a side effect of answering.

Two rules are enforced here, not merely requested in the prompt:

* Write tools pass through the memory stores' credential guard, so a secret
  pasted into a preference or fact is refused.
* Preference writes reject keys that duplicate engine-authoritative financial
  config, so memory can never shadow financial truth.
"""
from __future__ import annotations

import datetime
from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.models import User
from app.agent.tools.base import BaseTool
from app.memory import (
    MemoryServiceError,
    get_preferences,
    list_goals,
    save_fact,
    save_goal,
    save_preference,
    search_facts,
)
from app.memory.snapshot import fact_to_json, goal_to_json


def _error(exc: MemoryServiceError) -> dict[str, Any]:
    """Domain errors become structured results the model can act on."""
    return {"success": False, "error_code": exc.code, "message": exc.message}


class GetPreferencesArgs(BaseModel):
    pass


class GetPreferencesTool(BaseTool):
    name = "get_preferences"
    description = (
        "Read the user's stored soft preferences (tone, risk tolerance, display "
        "and notification choices). These are user-stated context only: never "
        "use them as financial facts."
    )
    args_schema = GetPreferencesArgs

    def execute(self, db: Session, user: User, args: dict[str, Any], as_of: date) -> Any:
        return {"success": True, "preferences": get_preferences(db, user)}


class SavePreferenceArgs(BaseModel):
    key: str = Field(..., description="Preference name, e.g. 'risk_tolerance'.")
    value: Any = Field(
        ...,
        description=(
            "The preference value (string, number, boolean, list, or object). "
            "Never a credential, API key, or token."
        ),
    )


class SavePreferenceTool(BaseTool):
    name = "save_preference"
    description = (
        "Save a soft, user-stated setting to durable memory. Use only when the "
        "user expresses a preference. Never store financial figures, balances, "
        "credentials, or anything that belongs in the financial profile."
    )
    args_schema = SavePreferenceArgs

    def execute(self, db: Session, user: User, args: dict[str, Any], as_of: date) -> Any:
        try:
            row = save_preference(db, user, key=args["key"], value=args["value"])
        except MemoryServiceError as exc:
            return _error(exc)
        return {"success": True, "key": row.key, "message": f"Saved preference {row.key!r}."}


class ListGoalsArgs(BaseModel):
    include_achieved: bool = Field(
        True, description="Whether to include goals already marked achieved."
    )


class ListGoalsTool(BaseTool):
    name = "list_goals"
    description = (
        "List the user's stated financial goals (saving, payoff, purchase). "
        "Goals are intentions, not obligations - they do not affect forecasts or "
        "affordability, which come from the engine."
    )
    args_schema = ListGoalsArgs

    def execute(self, db: Session, user: User, args: dict[str, Any], as_of: date) -> Any:
        goals = list_goals(db, user, include_achieved=bool(args.get("include_achieved", True)))
        return {"success": True, "goals": [goal_to_json(goal) for goal in goals]}


class SaveGoalArgs(BaseModel):
    kind: str = Field(..., description="Goal kind: 'save', 'payoff', or 'purchase'.")
    target_amount_minor: int = Field(
        ..., description="Target amount in minor units (e.g. cents). Must be positive."
    )
    target_date: Optional[datetime.date] = Field(
        None, description="Optional date by which the user hopes to reach the goal."
    )
    priority: int = Field(0, description="Relative priority; higher sorts first.")
    description: str = Field("", description="Short free-text description of the goal.")


class SaveGoalTool(BaseTool):
    name = "save_goal"
    description = (
        "Record a goal the user has stated. Use only for a goal the user "
        "actually expressed - never to invent a target, and never as a way to "
        "schedule a payment (payments are transactions)."
    )
    args_schema = SaveGoalArgs

    def execute(self, db: Session, user: User, args: dict[str, Any], as_of: date) -> Any:
        try:
            goal = save_goal(
                db,
                user,
                kind=args["kind"],
                target_amount_minor=args["target_amount_minor"],
                target_date=args.get("target_date"),
                priority=args.get("priority", 0),
                description=args.get("description", ""),
            )
        except MemoryServiceError as exc:
            return _error(exc)
        return {"success": True, "goal": goal_to_json(goal)}


class SearchFactsArgs(BaseModel):
    query_text: str = Field("", description="Substring to match; empty returns all facts.")
    as_of: Optional[datetime.date] = Field(
        None, description="Only return facts valid on this date. Defaults to today."
    )


class SearchFactsTool(BaseTool):
    name = "search_facts"
    description = (
        "Search dated facts the user has stated (e.g. 'contract ends 2027-03'). "
        "These are user statements for context, not financial records."
    )
    args_schema = SearchFactsArgs

    def execute(self, db: Session, user: User, args: dict[str, Any], as_of: date) -> Any:
        effective = args.get("as_of") or as_of
        facts = search_facts(db, user, query_text=args.get("query_text", ""), as_of=effective)
        return {"success": True, "facts": [fact_to_json(fact) for fact in facts]}


class SaveFactArgs(BaseModel):
    text: str = Field(..., description="The fact, in the user's own words.")
    valid_from: Optional[datetime.date] = Field(
        None, description="Date the fact starts applying (optional)."
    )
    valid_to: Optional[datetime.date] = Field(
        None, description="Date the fact stops applying (optional)."
    )


class SaveFactTool(BaseTool):
    name = "save_fact"
    description = (
        "Store a durable fact the user stated about their circumstances. Use only "
        "for something the user actually said. Never store balances, amounts owed, "
        "or credentials - financial figures come from the ledger and the engine."
    )
    args_schema = SaveFactArgs

    def execute(self, db: Session, user: User, args: dict[str, Any], as_of: date) -> Any:
        try:
            # Convert string dates to date objects if necessary
            valid_from = args.get("valid_from")
            if isinstance(valid_from, str):
                valid_from = date.fromisoformat(valid_from)
            valid_to = args.get("valid_to")
            if isinstance(valid_to, str):
                valid_to = date.fromisoformat(valid_to)

            fact = save_fact(
                db,
                user,
                text=args["text"],
                valid_from=valid_from,
                valid_to=valid_to,
            )
        except MemoryServiceError as exc:
            return _error(exc)
        return {"success": True, "fact": fact_to_json(fact)}

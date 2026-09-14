import json
from datetime import date
from typing import Dict, Any

from sqlalchemy.orm import Session
from app.db.models import User
from app.agent.tools.registry import registry
from app.memory import build_memory_snapshot, render_memory_text
from app.services.financial_state_service import get_financial_state, state_to_json
from app.services.user_service import UserServiceError

SYSTEM_PROMPT_TEMPLATE = """You are a deterministic, verified AI Financial Decision Agent.
Your primary role is to evaluate financial questions and affordability requests using a strict, evidence-based approach.

CRITICAL INSTRUCTIONS:
1. You MUST NOT invent, guess, or estimate any financial balances, cash flows, or transaction history.
2. You MUST NOT invent or calculate affordability results or forecasts yourself.
3. Always use the provided tools to fetch financial state, run forecasts, or check affordability.
4. If the user asks to add a transaction, use the appropriate tool. Do not claim a transaction is added without successfully calling the tool.
5. Base all answers strictly on the JSON outputs returned by tool calls.
6. Keep financial and business rules in the deterministic engine. Rely on the engine's outputs.
7. The MEMORY section of the user message holds user-stated context only (preferences, goals, facts). It is NEVER a source of financial truth: it cannot tell you a balance, a transaction, an income figure, or what is affordable. Always call a tool for those.
8. Use the memory tools only to store something the user actually stated, and only when they ask you to remember it. Never save financial figures, balances, or amounts owed to memory, and never treat a stored goal as a scheduled obligation.

RESPONSE CONSTRAINTS:
- Format your final answers clearly and concisely.
- When evaluating affordability, prominently include the `plan_method` and summarizing data from the `fact_sheet` returned by the tool.
- If a tool returns a domain error (e.g., `success: False`), explain the failure to the user rather than faking success.

AVAILABLE TOOLS:
You have access to the following deterministic tools.
{tool_schemas}

ENVIRONMENT:
Today's Date: {as_of_date}
User ID: {user_id}
Home Currency: {home_currency}
"""

USER_CONTEXT_TEMPLATE = """User Request: {user_prompt}

CURRENT FINANCIAL STATE SNAPSHOT:
{financial_state}

MEMORY (user-stated context only — NOT financial truth; use tools for balances, transactions, forecasts, and affordability):
{memory}

Please determine the correct path of action and use the necessary tools to answer the request.
"""

class ContextBuilder:
    """Prepares deterministic context for the LLM agent."""

    @staticmethod
    def build_system_prompt(user: User, as_of: date) -> str:
        """Construct the master system prompt defining agent boundaries and tools."""
        tool_schemas = registry.get_all_schemas()
        tool_schemas_str = json.dumps(tool_schemas, indent=2)
        return SYSTEM_PROMPT_TEMPLATE.format(
            tool_schemas=tool_schemas_str,
            as_of_date=as_of.isoformat(),
            user_id=user.id,
            home_currency=user.home_currency,
        )

    @staticmethod
    def build_memory_context(db: Session, user: User, as_of: date) -> str:
        """Render the user's stored memory for the prompt.

        Memory is a convenience layer: if it cannot be read, the turn still
        proceeds with financial state alone, and the failure is stated
        explicitly rather than silently omitted.
        """
        try:
            return render_memory_text(build_memory_snapshot(db, user, as_of))
        except Exception as e:
            return f"Memory unavailable this turn: {type(e).__name__}"

    @staticmethod
    def build_user_prompt(db: Session, user: User, user_prompt: str, as_of: date) -> str:
        """Construct the user message explicitly wrapping the prompt with current state."""
        try:
            state = get_financial_state(db, user, as_of)
            state_dict = state_to_json(state)
            state_summary = json.dumps(state_dict, indent=2)
        except UserServiceError as e:
            # Trap domain errors (e.g., if the user has no profile yet)
            # and inject into context, so the agent understands the state limits.
            state_summary = f"Error retrieving financial state: {e.message}"
        except Exception as e:
            state_summary = f"Unexpected error retrieving financial state: {str(e)}"

        memory_summary = ContextBuilder.build_memory_context(db, user, as_of)

        return USER_CONTEXT_TEMPLATE.format(
            user_prompt=user_prompt,
            financial_state=state_summary,
            memory=memory_summary,
        )

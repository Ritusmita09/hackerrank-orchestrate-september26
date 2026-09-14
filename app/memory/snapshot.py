"""Memory snapshot — the deterministic read the agent context is built from.

`build_memory_snapshot` collects one user's preferences, active goals, and
facts valid at ``as_of``. It is a *read*: nothing here writes, and nothing here
consults the ledger.

The snapshot is rendered to plain text for prompt injection. Two properties
matter and are tested:

* **Deterministic** — the same stored memory always renders byte-identically,
  because every collection is ordered (preferences by key, goals by priority
  then creation, facts by creation then id).
* **No braces** — nested values are rendered as ``key=value`` pairs rather
  than JSON objects, so the memory block cannot be mistaken for, or corrupt,
  the financial-state JSON that precedes it in the prompt.

The snapshot never carries balances, transactions, forecasts, or affordability
results. Those come from the tools and the engine alone.

**Bounded by design.** The rendered block is injected into *every* user prompt,
and the orchestrator re-sends its whole message history on each step, so an
unbounded render would inflate every later request. The renderer therefore
applies a fixed context budget: at most ``MAX_*_IN_PROMPT`` entries per store,
each truncated to ``MAX_RENDERED_ENTRY_CHARS``, with an explicit "N more
omitted" line whenever something was dropped. The budget is deterministic -
collections are already deterministically ordered, so the same stored memory
always yields the same block - and lossless in reach: the stores themselves
still return everything, and the memory tools (``search_facts``,
``list_goals``) remain the complete read path.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, Sequence

from sqlalchemy.orm import Session

from ..db.models import FinancialFact, FinancialGoal, User
from .facts import list_active_facts
from .goals import list_goals
from .preferences import get_preferences

#: Shown when the user has no stored memory at all.
EMPTY_MEMORY_TEXT = "No stored memory for this user yet."

#: Context budget for the prompt block. Memory is a convenience layer, so a
#: fixed cap plus a pointer to the tools beats an unbounded injection.
MAX_PREFERENCES_IN_PROMPT = 30
MAX_GOALS_IN_PROMPT = 10
MAX_FACTS_IN_PROMPT = 20
MAX_RENDERED_ENTRY_CHARS = 300


def _clip(text: str, limit: int = MAX_RENDERED_ENTRY_CHARS) -> str:
    """Truncate one rendered entry, marking that it was cut."""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "... [truncated]"


def _escape_braces(text: str) -> str:
    """Neutralize braces in user-stated text.

    The block must stay brace-free so it can never be mistaken for, or corrupt,
    the financial-state JSON that precedes it in the prompt. Structured values
    are already brace-free by construction; free text is not, so a brace a user
    actually typed is emitted as a JSON-style escape. This affects the prompt
    summary only - the memory tools return the text exactly as stored.
    """
    return text.replace("{", "\\u007b").replace("}", "\\u007d")


def goal_to_json(goal: FinancialGoal) -> Dict[str, Any]:
    """Serialize one goal for a snapshot or tool result."""
    return {
        "goal_id": goal.id,
        "kind": goal.kind,
        "target_amount_minor": goal.target_amount_minor,
        "currency": goal.currency,
        "target_date": goal.target_date.isoformat() if goal.target_date else None,
        "priority": goal.priority,
        "description": goal.description or "",
        "achieved": goal.achieved_at is not None,
    }


def fact_to_json(fact: FinancialFact) -> Dict[str, Any]:
    """Serialize one fact for a snapshot or tool result."""
    return {
        "fact_id": fact.id,
        "text": fact.text,
        "source": fact.source,
        "valid_from": fact.valid_from.isoformat() if fact.valid_from else None,
        "valid_to": fact.valid_to.isoformat() if fact.valid_to else None,
    }


def build_memory_snapshot(db: Session, user: User, as_of: date) -> Dict[str, Any]:
    """Read the user's stored memory as a deterministic, JSON-safe dict.

    Goals that are already achieved are omitted — they are history, not
    context for the current turn. Facts outside their validity window at
    ``as_of`` are omitted for the same reason.
    """
    goals: Sequence[FinancialGoal] = list_goals(db, user, include_achieved=False)
    facts: Sequence[FinancialFact] = list_active_facts(db, user, as_of=as_of)
    return {
        "preferences": get_preferences(db, user),
        "goals": [goal_to_json(goal) for goal in goals],
        "facts": [fact_to_json(fact) for fact in facts],
    }


def has_content(snapshot: Dict[str, Any]) -> bool:
    """True when the snapshot holds anything worth putting in the prompt."""
    return bool(
        snapshot.get("preferences") or snapshot.get("goals") or snapshot.get("facts")
    )


def _render_value(value: Any) -> str:
    """Render a preference value without braces (see module docstring)."""
    if isinstance(value, dict):
        return "; ".join(f"{key}={_render_value(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(_render_value(item) for item in value)
    return str(value)


def render_memory_text(snapshot: Dict[str, Any]) -> str:
    """Render a snapshot as brace-free plain text for the prompt.

    Applies the module's fixed context budget (see the module docstring), so
    the block stays small no matter how much memory the user has accumulated.
    """
    if not has_content(snapshot):
        return EMPTY_MEMORY_TEXT

    lines: list[str] = []

    preferences = snapshot.get("preferences") or {}
    if preferences:
        lines.append("Preferences:")
        keys = sorted(preferences)
        for key in keys[:MAX_PREFERENCES_IN_PROMPT]:
            rendered = _escape_braces(_clip(_render_value(preferences[key])))
            lines.append(f"- {key}: {rendered}")
        if len(keys) > MAX_PREFERENCES_IN_PROMPT:
            lines.append(
                f"- ({len(keys) - MAX_PREFERENCES_IN_PROMPT} more preferences not "
                "shown; call get_preferences for the full set)"
            )

    goals = snapshot.get("goals") or []
    if goals:
        lines.append("Goals (user-stated intentions, not obligations):")
        for goal in goals[:MAX_GOALS_IN_PROMPT]:
            target = f"{goal['target_amount_minor']} {goal['currency']} minor units"
            deadline = f", target date {goal['target_date']}" if goal.get("target_date") else ""
            note = f" — {goal['description']}" if goal.get("description") else ""
            lines.append(_escape_braces(_clip(f"- [{goal['kind']}] {target}{deadline}{note}")))
        if len(goals) > MAX_GOALS_IN_PROMPT:
            lines.append(
                f"- ({len(goals) - MAX_GOALS_IN_PROMPT} more goals not shown; "
                "call list_goals for the full set)"
            )

    facts = snapshot.get("facts") or []
    if facts:
        lines.append("Facts the user has stated:")
        for fact in facts[:MAX_FACTS_IN_PROMPT]:
            text = fact["text"]
            window = ""
            if fact.get("valid_from") or fact.get("valid_to"):
                window = f" (valid {fact.get('valid_from') or 'any'} to {fact.get('valid_to') or 'any'})"
            lines.append(_escape_braces(_clip(f"- {text}{window}")))
        if len(facts) > MAX_FACTS_IN_PROMPT:
            lines.append(
                f"- ({len(facts) - MAX_FACTS_IN_PROMPT} more facts not shown; "
                "call search_facts for the full set)"
            )

    return "\n".join(lines)

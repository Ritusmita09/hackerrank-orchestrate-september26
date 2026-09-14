"""Memory stores — durable user context, never financial truth.

ARCHITECTURE.md §10 defines four stores behind one interface. This package
implements the durable ones over tables that already existed in the initial
schema:

=============  ==========================  ======================
Store          Content                     Table
=============  ==========================  ======================
Preferences    soft, user-stated settings  ``user_preferences``
Goals          savings/payoff intentions   ``financial_goals``
Facts          dated free-text statements  ``financial_facts``
=============  ==========================  ======================

Two boundaries are structural, not conventional:

1. **Memory is never a source of financial truth.** Balances, transactions,
   forecasts, affordability, income, and payment obligations come from the
   ledger, the services, and the deterministic engine — never from here. A
   goal is an intention, not a scheduled obligation; it never enters a
   forecast. Preferences that would duplicate authoritative financial config
   (the protected/reducible/stoppable sets, the minimum-balance floor) are
   refused, pointing the caller at the financial profile instead.
2. **Writes are explicit and audited.** Nothing an LLM says is persisted
   implicitly; only a deliberate call to a ``save_*`` function writes, and
   every write appends an ``audit_events`` row.

Credentials are refused outright (see :mod:`app.memory.secrets`).
"""
from .errors import MemoryServiceError
from .facts import (
    close_fact,
    get_fact,
    list_active_facts,
    save_fact,
    search_facts,
)
from .goals import get_goal, list_goals, save_goal, update_goal
from .preferences import (
    delete_preference,
    get_preference,
    get_preferences,
    save_preference,
)
from .snapshot import build_memory_snapshot, render_memory_text

__all__ = [
    "MemoryServiceError",
    # preferences
    "save_preference",
    "get_preference",
    "get_preferences",
    "delete_preference",
    # goals
    "save_goal",
    "get_goal",
    "list_goals",
    "update_goal",
    # facts
    "save_fact",
    "get_fact",
    "list_active_facts",
    "search_facts",
    "close_fact",
    # context assembly
    "build_memory_snapshot",
    "render_memory_text",
]

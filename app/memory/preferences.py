"""Preferences store — soft, user-stated settings.

Backed by the ``user_preferences`` table (one row per user+key, enforced by a
unique constraint, so a save is an upsert).

**What belongs here:** soft, advisory context the agent should respect when it
talks to the user — risk tolerance, preferred tone, categories the user cares
about, display and notification choices.

**What does not:** anything the deterministic engine treats as authoritative.
The protected/reducible/stoppable category sets, the minimum-balance floor,
and the payment-method willingness live on the versioned ``financial_profiles``
row and are consumed by the engine. Duplicating them here would create a
second source of truth for a financial rule — the exact failure this boundary
exists to prevent — so those keys are refused (``reserved_key``) with a
pointer to the profile service.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..db.models import User, UserPreference
from ..services.audit_service import record_audit_event
from .errors import MemoryServiceError
from .secrets import assert_storable

#: Preference keys refused because an authoritative home already exists.
#: Keyed by normalized key name -> where the truth actually lives.
RESERVED_KEYS: dict[str, str] = {
    "currentbalance": "financial profile (balances come from the engine)",
    "currentbalanceminor": "financial profile (balances come from the engine)",
    "balance": "financial profile / ledger (never memory)",
    "minimumbalance": "financial profile (the engine's hard floor)",
    "minimumbalanceminor": "financial profile (the engine's hard floor)",
    "minimumbalancetokeep": "financial profile (the engine's hard floor)",
    "protectedcategories": "financial profile (engine-authoritative authorization)",
    "reduciblecategories": "financial profile (engine-authoritative authorization)",
    "stoppablecategories": "financial profile (engine-authoritative authorization)",
    "paymentmethodsconsidered": "financial profile (engine-authoritative authorization)",
    "maxinstallmentpayments": "financial profile (engine-authoritative authorization)",
    "homecurrency": "the user record (set at account creation)",
    "safeamount": "the engine (compute_safe_amount)",
    "maxsafeamount": "the engine (compute_safe_amount)",
    "affordable": "the engine (check_affordability)",
}

MAX_KEY_LENGTH = 128


def _normalize_key(key: str) -> str:
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def _validate_key(key: str) -> str:
    cleaned = (key or "").strip()
    if not cleaned:
        raise MemoryServiceError("invalid_preference", "a preference key is required.")
    if len(cleaned) > MAX_KEY_LENGTH:
        raise MemoryServiceError(
            "invalid_preference", f"preference keys are limited to {MAX_KEY_LENGTH} characters."
        )
    reserved = RESERVED_KEYS.get(_normalize_key(cleaned))
    if reserved is not None:
        raise MemoryServiceError(
            "reserved_key",
            f"{cleaned!r} is not storable as a preference: that value is "
            f"authoritative in the {reserved}. Financial truth never comes "
            "from memory.",
        )
    return cleaned


def save_preference(db: Session, user: User, *, key: str, value: Any) -> UserPreference:
    """Create or update one preference. Explicit and audited.

    Raises:
        MemoryServiceError: the key is empty/oversized/reserved, or the value
            looks like a credential.
    """
    cleaned = _validate_key(key)
    assert_storable(cleaned, value)

    row = db.scalar(
        select(UserPreference)
        .where(UserPreference.user_id == user.id)
        .where(UserPreference.key == cleaned)
    )
    created = row is None
    if row is None:
        row = UserPreference(user_id=user.id, key=cleaned, value=value)
        db.add(row)
    else:
        row.value = value

    record_audit_event(
        db,
        event_type="memory_preference_saved",
        user_id=user.id,
        payload={"key": cleaned, "created": created},
    )
    db.flush()
    return row


def get_preference(db: Session, user: User, key: str) -> Any | None:
    """One preference's value, or None. Scoped to the user."""
    row = db.scalar(
        select(UserPreference)
        .where(UserPreference.user_id == user.id)
        .where(UserPreference.key == (key or "").strip())
    )
    return None if row is None else row.value


def get_preferences(db: Session, user: User) -> dict[str, Any]:
    """Every preference for the user, ordered by key for deterministic output."""
    rows = db.scalars(
        select(UserPreference)
        .where(UserPreference.user_id == user.id)
        .order_by(UserPreference.key)
    ).all()
    return {row.key: row.value for row in rows}


def delete_preference(db: Session, user: User, key: str) -> bool:
    """Remove one preference. Returns whether a row was removed."""
    cleaned = (key or "").strip()
    row = db.scalar(
        select(UserPreference)
        .where(UserPreference.user_id == user.id)
        .where(UserPreference.key == cleaned)
    )
    if row is None:
        return False

    db.execute(
        delete(UserPreference)
        .where(UserPreference.user_id == user.id)
        .where(UserPreference.key == cleaned)
    )
    record_audit_event(
        db, event_type="memory_preference_deleted", user_id=user.id, payload={"key": cleaned}
    )
    db.flush()
    return True

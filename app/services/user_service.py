"""User and profile management.

The financial profile is versioned: every update appends a new revision
(append-oriented history) and state reconstruction always reads the latest
revision, so past decisions remain attributable to the profile they used.
"""
from __future__ import annotations

from datetime import date
from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.models import Money, PaymentMethod, UserFinancialProfile
from ..db.models import FinancialProfile, User
from .audit_service import record_audit_event


class UserServiceError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


DEFAULT_PAYMENT_METHODS: tuple[str, ...] = ("full_payment",)


def create_user(
    db: Session,
    *,
    email: str,
    display_name: str = "",
    home_currency: str = "USD",
    current_balance_minor: int = 0,
    minimum_balance_minor: int = 0,
    protected_categories: Sequence[str] = (),
    reducible_categories: Sequence[str] = (),
    stoppable_categories: Sequence[str] = (),
    payment_methods_considered: Sequence[str] = DEFAULT_PAYMENT_METHODS,
    max_installment_payments: Optional[int] = None,
) -> User:
    """Create a user together with their first financial-profile revision."""
    email = email.strip().lower()
    existing = db.scalar(select(User).where(User.email == email))
    if existing is not None:
        raise UserServiceError("email_taken", f"a user with email {email!r} already exists")

    user = User(email=email, display_name=display_name.strip(), home_currency=home_currency)
    db.add(user)
    db.flush()  # assign user.id before the profile references it

    profile = FinancialProfile(
        user_id=user.id,
        revision=1,
        current_balance_minor=current_balance_minor,
        minimum_balance_minor=minimum_balance_minor,
        currency=home_currency,
        protected_categories=list(protected_categories),
        reducible_categories=list(reducible_categories),
        stoppable_categories=list(stoppable_categories),
        payment_methods_considered=list(payment_methods_considered),
        max_installment_payments=max_installment_payments,
    )
    db.add(profile)
    record_audit_event(
        db, event_type="user_created",
        user_id=user.id,
        payload={"email": email, "home_currency": home_currency},
    )
    db.flush()
    return user


def get_user(db: Session, user_id: str) -> User:
    user = db.get(User, user_id)
    if user is None:
        raise UserServiceError("user_not_found", f"user {user_id!r} does not exist")
    return user


def get_latest_profile(db: Session, user_id: str) -> FinancialProfile:
    """The newest financial-profile revision for the user."""
    profile = db.scalar(
        select(FinancialProfile)
        .where(FinancialProfile.user_id == user_id)
        .order_by(FinancialProfile.revision.desc())
        .limit(1)
    )
    if profile is None:
        raise UserServiceError(
            "profile_not_found", f"user {user_id!r} has no financial profile"
        )
    return profile


def update_profile(
    db: Session,
    user_id: str,
    *,
    current_balance_minor: Optional[int] = None,
    minimum_balance_minor: Optional[int] = None,
    protected_categories: Optional[Sequence[str]] = None,
    reducible_categories: Optional[Sequence[str]] = None,
    stoppable_categories: Optional[Sequence[str]] = None,
    payment_methods_considered: Optional[Sequence[str]] = None,
    max_installment_payments: Optional[int] = None,
) -> FinancialProfile:
    """Append a new profile revision; earlier revisions stay untouched."""
    latest = get_latest_profile(db, user_id)

    def _or_current(new, current):
        return current if new is None else new

    revision = FinancialProfile(
        user_id=user_id,
        revision=latest.revision + 1,
        current_balance_minor=_or_current(current_balance_minor, latest.current_balance_minor),
        minimum_balance_minor=_or_current(minimum_balance_minor, latest.minimum_balance_minor),
        currency=latest.currency,
        protected_categories=list(_or_current(protected_categories, latest.protected_categories)),
        reducible_categories=list(_or_current(reducible_categories, latest.reducible_categories)),
        stoppable_categories=list(_or_current(stoppable_categories, latest.stoppable_categories)),
        payment_methods_considered=list(
            _or_current(payment_methods_considered, latest.payment_methods_considered)
        ),
        max_installment_payments=_or_current(max_installment_payments, latest.max_installment_payments),
    )
    db.add(revision)
    record_audit_event(
        db, event_type="profile_revised", user_id=user_id,
        payload={"revision": revision.revision},
    )
    db.flush()
    return revision


def profile_to_engine_profile(
    profile: FinancialProfile, user_id: str, as_of: date
) -> UserFinancialProfile:
    """Translate the latest DB revision into the engine's typed profile."""
    currency = profile.currency
    methods = frozenset(
        PaymentMethod(m) for m in profile.payment_methods_considered or ["full_payment"]
    )
    return UserFinancialProfile(
        user_id=user_id,
        home_currency=currency,
        current_available_balance=Money.from_minor(profile.current_balance_minor, currency),
        minimum_balance_to_keep=Money.from_minor(profile.minimum_balance_minor, currency),
        protected_categories=frozenset(profile.protected_categories or ()),
        reducible_categories=frozenset(profile.reducible_categories or ()),
        stoppable_categories=frozenset(profile.stoppable_categories or ()),
        payment_methods_considered=methods,
        max_installment_payments=profile.max_installment_payments,
    )

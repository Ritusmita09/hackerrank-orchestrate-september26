"""Financial State Reconstruction.

Ingests heterogeneous historical/future events and amendments into a strictly
ordered, conflict-resolved ledger, then slices it into the exact
Classification groups the simulator needs.

Replaces the old system's brittle message-keyword heuristics with explicit
``ConflictResolution`` logic and separation of concerns.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

from .models import (
    ConflictResolution,
    Direction,
    EventStatus,
    ExpenseStream,
    FinancialEvent,
    FinancialState,
    IncomeStream,
    RecurringPattern,
    UserFinancialProfile,
)


@dataclass(frozen=True)
class BuildContext:
    """The raw ingredients before state compilation."""

    user_id: str
    as_of: date
    profile: UserFinancialProfile
    events: Sequence[FinancialEvent]
    patterns: Sequence[RecurringPattern]


def _require_currency(events: Sequence[FinancialEvent], expected_currency: str) -> None:
    for event in events:
        if event.amount is None:
            continue
        if event.amount.currency != expected_currency:
            from .models import CurrencyNotNormalizedError
            raise CurrencyNotNormalizedError(
                f"event {event.event_id} has currency {event.amount.currency}, "
                f"expected user's home currency {expected_currency}. The "
                f"ingestion pipeline must normalize currencies before "
                f"passing events to the engine."
            )


def build_state(context: BuildContext) -> FinancialState:
    """Reconstruct a complete FinancialState from raw, un-deduplicated inputs.

    Classification rules:
    - Settled events before ``as_of`` → history.
    - Scheduled events at/after ``as_of`` → future_confirmed.
    - Pending debits → pending_debits (reserved at forecast start).
    - Pending credits → pending_credits (kept in state, but excluded from forecasts).
    - Cancelled / failed / unrealized → excluded entirely.

    Deduplication rules:
    - Exactly one event may claim a (stream_key, date, direction) slot.
    - Rule 1: an explicit amendment wins over the amended event.
    - Rule 2: SETTLED wins over SCHEDULED or PENDING.
    - Rule 3: mathematically safer interpretation wins (smaller credit, larger debit).

    Raises:
        CurrencyNotNormalizedError: if any event's currency differs from
            ``profile.home_currency``.
        MissingAmountError: if any flow-relevant event has a blank amount.
    """
    _require_currency(context.events, context.profile.home_currency)

    # 1. Filter out non-cash flows and check amounts.
    flow_events: List[FinancialEvent] = []
    for event in context.events:
        if event.status in (EventStatus.CANCELLED, EventStatus.FAILED,
                            EventStatus.UNREALIZED, EventStatus.FORECASTED):
            continue
        if event.amount is None:
            from .models import MissingAmountError
            raise MissingAmountError(
                f"flow-relevant event {event.event_id} (status={event.status}) "
                f"has no amount"
            )
        flow_events.append(event)

    # 2. Group by conflict slot: (stream_key, date, direction)
    groups: Dict[Tuple[str, date, Direction], List[FinancialEvent]] = {}
    for event in flow_events:
        slot = (event.stream_key(), event.date, event.direction)
        groups.setdefault(slot, []).append(event)

    # 3. Resolve conflicts
    resolved: List[FinancialEvent] = []
    conflict_log: List[ConflictResolution] = []

    for slot, group in groups.items():
        if len(group) == 1:
            resolved.append(group[0])
            continue

        # Sort to resolve (Rule 1: explicit amendments)
        amends_ids = {e.amends_event_id for e in group if e.amends_event_id}
        kept_event: Optional[FinancialEvent] = None
        rule = ""
        # Drop anything explicitly superseded inside this group
        filtered = [e for e in group if e.event_id not in amends_ids]

        if len(filtered) == 1:
            kept_event = filtered[0]
            rule = "explicit_amendment"
        else:
            # Rule 2: Status precedence
            has_settled = any(e.status is EventStatus.SETTLED for e in filtered)
            if has_settled:
                filtered = [e for e in filtered if e.status is EventStatus.SETTLED]
                if len(filtered) == 1:
                    kept_event = filtered[0]
                    rule = "status_precedence"

            # Rule 3: Safest amount (if still tied)
            if kept_event is None:
                if slot[2] is Direction.CREDIT:
                    # Smallest amount for income
                    kept_event = min(filtered, key=lambda e: e.amount.amount_minor)
                else:
                    # Largest amount for expense
                    kept_event = max(filtered, key=lambda e: e.amount.amount_minor)
                rule = "financially_safer"

        resolved.append(kept_event)
        dropped = [e.event_id for e in group if e.event_id != kept_event.event_id]
        conflict_log.append(ConflictResolution(
            stream_key=slot[0], date=slot[1], direction=slot[2],
            kept_event_id=kept_event.event_id, dropped_event_ids=tuple(dropped),
            rule=rule
        ))

    # 4. Classify the resolved, deduplicated events.
    history: List[FinancialEvent] = []
    future_confirmed: List[FinancialEvent] = []
    pending_debits: List[FinancialEvent] = []
    pending_credits: List[FinancialEvent] = []

    for event in resolved:
        if event.status is EventStatus.SETTLED and event.date < context.as_of:
            history.append(event)
        elif event.status is EventStatus.SCHEDULED and event.date >= context.as_of:
            future_confirmed.append(event)
        elif event.status is EventStatus.PENDING:
            if event.direction is Direction.DEBIT:
                pending_debits.append(event)
            else:
                pending_credits.append(event)
        # Pending/scheduled events before as_of are orphaned — they should have
        # settled. The engine ignores them under the assumption that if they
        # had moved real money, the bank import would have caught them as SETTLED.

    # 5. Build high-level streams
    patterns_by_key = {p.stream_key: p for p in context.patterns}
    stream_keys = {e.stream_key() for e in history} | \
                  {e.stream_key() for e in future_confirmed} | \
                  set(patterns_by_key.keys())

    income_streams: List[IncomeStream] = []
    expense_streams: List[ExpenseStream] = []

    for key in sorted(stream_keys):
        stream_history = tuple(e for e in history if e.stream_key() == key)
        stream_future = tuple(e for e in future_confirmed if e.stream_key() == key)
        pattern = patterns_by_key.get(key)
        # Derive stream-level metadata from its most recent event, fallback to pattern
        components = sorted(list(stream_history) + list(stream_future),
                            key=lambda e: (e.date, e.event_id))
        if not components and pattern is None:
            continue
        category = components[-1].category if components else pattern.category
        description = components[-1].description if components else pattern.description
        direction = components[-1].direction if components else pattern.direction

        if direction is Direction.CREDIT:
            income_streams.append(IncomeStream(
                stream_key=key, category=category, description=description,
                history=stream_history, future_confirmed=stream_future,
                pattern=pattern
            ))
        else:
            flexibility = components[-1].flexibility if components else pattern.flexibility
            # The most recent event's minimum wins when set; otherwise fall
            # back to the pattern's, so a declared pattern minimum is not
            # silently lost for streams that also have history.
            minimum = None
            if components and components[-1].min_allowed_amount is not None:
                minimum = components[-1].min_allowed_amount
            elif pattern is not None:
                minimum = pattern.min_allowed_amount
            expense_streams.append(ExpenseStream(
                stream_key=key, category=category, description=description,
                history=stream_history, future_confirmed=stream_future,
                pattern=pattern, flexibility=flexibility,
                min_allowed_amount=minimum,
                is_protected=category in context.profile.protected_categories,
                can_reduce=category in context.profile.reducible_categories,
                can_stop=category in context.profile.stoppable_categories
            ))

    # 6. Sort chronological tuples
    return FinancialState(
        user_id=context.user_id,
        as_of=context.as_of,
        home_currency=context.profile.home_currency,
        current_balance=context.profile.current_available_balance,
        profile=context.profile,
        history=tuple(sorted(history, key=lambda e: (e.date, e.event_id))),
        future_confirmed=tuple(sorted(future_confirmed, key=lambda e: (e.date, e.event_id))),
        pending_debits=tuple(sorted(pending_debits, key=lambda e: (e.date, e.event_id))),
        pending_credits=tuple(sorted(pending_credits, key=lambda e: (e.date, e.event_id))),
        patterns=tuple(context.patterns),
        income_streams=tuple(income_streams),
        expense_streams=tuple(expense_streams),
        conflict_log=tuple(conflict_log)
    )

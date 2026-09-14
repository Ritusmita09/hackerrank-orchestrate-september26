"""Recurrence detection over the event ledger.

Ports the sound concepts from the reference project and fixes its structural
flaws:

* **Stream-key grouping** — events group by (normalized description,
  category, direction). The old system grouped by (category, direction,
  flexibility), which merged two different salary streams into one broken
  pattern. Here, distinct streams produce distinct patterns or none at all.
* **No keyword heuristics.** The old system special-cased "salary",
  "bonus", "arrears" via substring matching on descriptions — grader-shaped
  logic. Here a one-off adjustment simply forms its own one-event stream,
  which never reaches the minimum occurrence count, so no pattern is
  proposed. Purely structural.
* **Integer estimation.** Typical amounts are computed with integer math
  (nearest-rank percentiles, conservative median rounding) — no floats touch
  money, even in estimation.
* **Conservative by direction.** Income estimates round down (never
  overstate available money); expense estimates round up (never understate
  obligations).
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

from .models import (
    Direction,
    Estimator,
    EventStatus,
    FinancialEvent,
    FixedPeriod,
    Flexibility,
    Money,
    MonthlyPeriod,
    Provenance,
    RecurringPattern,
    SourceType,
)


@dataclass(frozen=True)
class DetectionConfig:
    """Tunable detection policy. All knobs explicit, no ambient constants."""

    min_occurrences: int = 3
    interval_tolerance: int = 2          # days an interval may deviate from a period
    interval_match_ratio: float = 0.7    # fraction of intervals that must match
    day_of_month_dominance: float = 0.6  # dominance needed for a monthly anchor
    semi_monthly_balance: float = 0.4    # balance needed between two anchor days
    common_periods: Tuple[int, ...] = (7, 10, 13, 14, 15, 21, 28, 29, 30, 31)
    #: Income patterns below this confidence are excluded from the state
    #: (forecasting income that may not arrive is unsafe). Expense patterns
    #: are kept at ANY confidence: over-forecasting expenses is the safe error.
    income_confidence_threshold: float = 0.5


DEFAULT_DETECTION_CONFIG = DetectionConfig()


def _sorted_amounts_minor(events: Sequence[FinancialEvent]) -> List[int]:
    return sorted(e.amount.amount_minor for e in events if e.amount is not None)


def _nearest_rank_percentile(sorted_values: List[int], fraction: float) -> int:
    """Nearest-rank percentile using pure integer math (no interpolation)."""
    if not sorted_values:
        raise ValueError("percentile of empty sequence")
    rank = max(1, math.ceil(fraction * len(sorted_values)))
    return sorted_values[rank - 1]


def _conservative_median(sorted_values: List[int], round_down: bool) -> int:
    """Median of integers; a .5 midpoint rounds down (income) or up (expense)."""
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 1:
        return sorted_values[mid]
    total = sorted_values[mid - 1] + sorted_values[mid]
    return total // 2 if round_down else -((-total) // 2)


def typical_amount_for(events: Sequence[FinancialEvent], currency: str) -> Tuple[Money, Estimator]:
    """Estimate the typical amount for a stream, conservatively by direction.

    Income (credits): minimum if fewer than 10 samples, else the 10th
    percentile — never assume the good months. Expenses (debits): the
    median, rounded up. Both rounded in the safe direction.
    """
    values = _sorted_amounts_minor(events)
    if not values:
        raise ValueError("cannot estimate a typical amount from amount-less events")
    if events[0].direction is Direction.CREDIT:
        if len(values) >= 10:
            minor = _nearest_rank_percentile(values, 0.1)
            estimator = Estimator.P10
        else:
            minor = values[0]
            estimator = Estimator.MIN
        return Money.from_minor(minor, currency), estimator
    minor = _conservative_median(values, round_down=False)
    return Money.from_minor(minor, currency), Estimator.MEDIAN


class RecurrenceDetector:
    """Detects recurring-stream proposals from the event ledger."""

    def __init__(self, config: DetectionConfig = DEFAULT_DETECTION_CONFIG) -> None:
        self.config = config

    # -- public API -----------------------------------------------------------

    def detect(
        self,
        history: Sequence[FinancialEvent],
        future: Sequence[FinancialEvent] = (),
    ) -> Tuple[RecurringPattern, ...]:
        """Detect patterns from settled history, optionally seeded by future events.

        Future (scheduled) events only *seed* detection for their stream —
        they strengthen evidence and extend ``last_seen`` but never create a
        pattern for a stream with no history (one confirmed future event is
        not proof of recurrence).

        Only settled events may seed history-based detection; the caller is
        responsible for passing classified events (see state.build_state).
        """
        groups: Dict[str, List[FinancialEvent]] = defaultdict(list)
        for event in history:
            groups[event.stream_key()].append(event)
        future_by_key: Dict[str, List[FinancialEvent]] = defaultdict(list)
        for event in future:
            future_by_key[event.stream_key()].append(event)

        patterns: List[RecurringPattern] = []
        # Iterate keys in sorted order so pattern ordering is deterministic.
        for key in sorted(groups):
            pattern = self._detect_stream(key, groups[key], future_by_key.get(key, []))
            if pattern is not None:
                patterns.append(pattern)
        return tuple(patterns)

    # -- internals --------------------------------------------------------------

    def _detect_stream(
        self,
        key: str,
        history: List[FinancialEvent],
        seed: List[FinancialEvent],
    ) -> Optional[RecurringPattern]:
        combined = sorted(list(history) + list(seed), key=lambda e: (e.date, e.event_id))
        if len(combined) < self.config.min_occurrences:
            return None

        intervals = [
            (combined[i + 1].date - combined[i].date).days
            for i in range(len(combined) - 1)
        ]
        if not intervals:
            return None

        for period_days in self.config.common_periods:
            matched = [
                interval for interval in intervals
                if abs(interval - period_days) <= self.config.interval_tolerance
            ]
            if len(matched) < self.config.interval_match_ratio * len(intervals):
                continue

            period = self._classify_period(period_days, combined)
            last = combined[-1]
            currency = last.amount.currency if last.amount else None
            if currency is None:
                return None
            typical, estimator = typical_amount_for(combined, currency)
            confidence = round(len(matched) / len(intervals), 4)
            pattern_id = "pat-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
            return RecurringPattern(
                pattern_id=pattern_id,
                stream_key=key,
                category=last.category,
                direction=last.direction,
                period=period,
                typical_amount=typical,
                estimator=estimator,
                last_seen=last.date,
                evidence_event_ids=tuple(e.event_id for e in combined),
                confidence=confidence,
                user_confirmed=False,
                description=last.description,
                flexibility=last.flexibility,
                min_allowed_amount=last.min_allowed_amount,
            )
        return None

    def _classify_period(
        self, period_days: int, events: List[FinancialEvent]
    ) -> object:
        """Decide whether a ~N-day period is truly fixed-length or calendar-anchored.

        Biweekly-ish (13-15 day) streams that consistently hit two specific
        days of the month become semi-monthly (two anchors); monthly-ish
        (28-31 day) streams with a dominant day of the month become monthly
        (one anchor). Everything else stays a fixed-length period.
        """
        if period_days in (13, 14, 15) and len(events) >= 4:
            days = [e.date.day for e in events]
            distinct = sorted(set(days))
            if len(distinct) == 2:
                day1, day2 = distinct
                smallest = min(days.count(day1), days.count(day2))
                if smallest >= self.config.semi_monthly_balance * len(days):
                    return MonthlyPeriod((day1, day2))
        if period_days in (28, 29, 30, 31):
            month_days = [e.date.day for e in events]
            mode_day = max(set(month_days), key=month_days.count)
            if month_days.count(mode_day) >= self.config.day_of_month_dominance * len(month_days):
                return MonthlyPeriod((mode_day,))
        return FixedPeriod(period_days)


def pattern_occurrences(
    pattern: RecurringPattern, start: date, end: date
) -> Tuple[FinancialEvent, ...]:
    """Generate a pattern's future occurrences in ``[start, end]``.

    Occurrences are strictly after ``last_seen`` (the historical event itself
    is never regenerated) and carry PATTERN provenance with the pattern's
    confidence, so every forecasted number is traceable to its evidence.
    """
    dates = pattern.period.occurrences(start, end, pattern.last_seen)
    return tuple(
        FinancialEvent(
            event_id=f"forecast:{pattern.pattern_id}:{day.isoformat()}",
            date=day,
            direction=pattern.direction,
            amount=pattern.typical_amount,
            category=pattern.category,
            description=pattern.description,
            status=EventStatus.FORECASTED,
            flexibility=pattern.flexibility,
            min_allowed_amount=pattern.min_allowed_amount,
            provenance=Provenance(
                source_type=SourceType.PATTERN,
                source_id=pattern.pattern_id,
                confidence=pattern.confidence,
                confirmed_by_user=pattern.user_confirmed,
            ),
        )
        for day in dates
    )

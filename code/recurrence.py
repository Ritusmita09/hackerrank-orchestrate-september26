"""
Recurrence detection for Buy or Wait? challenge.

Detects recurring patterns in financial events.
"""
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import statistics


class RecurringPattern:
    """Represents a detected recurring pattern."""

    def __init__(self, category: str, period_days: int, typical_amount: float,
                 last_date: datetime, event_ids: List[str], flexibility: str = 'fixed',
                 minimum_allowed_amount: Optional[float] = None, day_of_month: Optional[int] = None):
        self.category = category
        self.period_days = period_days
        self.typical_amount = typical_amount
        self.last_date = last_date
        self.event_ids = event_ids
        self.flexibility = flexibility
        self.minimum_allowed_amount = minimum_allowed_amount
        # For monthly patterns (period_days == 0), days_of_month is the list of
        # anchor days (1-31) each month — one day for monthly, two for semi-monthly.
        if isinstance(day_of_month, (list, tuple)):
            self.days_of_month = sorted(day_of_month)
        elif day_of_month is not None:
            self.days_of_month = [day_of_month]
        else:
            self.days_of_month = None
        # Kept for backwards compatibility (first anchor day, or the raw value)
        self.day_of_month = day_of_month

    def next_occurrence(self, after_date: datetime) -> datetime:
        """Calculate next occurrence on or after given date."""
        if self.period_days == 0:
            return self._next_monthly_occurrence(after_date)

        # Fixed period pattern
        if after_date <= self.last_date:
            return self.last_date + timedelta(days=self.period_days)

        # Calculate how many periods have passed
        days_since = (after_date - self.last_date).days
        periods_passed = days_since // self.period_days

        # Calculate the next occurrence
        next_date = self.last_date + timedelta(days=self.period_days * periods_passed)

        # If that's before the after_date, add one more period
        if next_date < after_date:
            next_date += timedelta(days=self.period_days)

        return next_date

    def _next_monthly_occurrence(self, after_date: datetime) -> datetime:
        """
        Calculate next occurrence on or after given date for a monthly pattern.

        Supports one anchor day per month (monthly) or two (semi-monthly).
        """
        import calendar

        days = self.days_of_month or ([self.day_of_month] if self.day_of_month else [self.last_date.day])

        year, month = self.last_date.year, self.last_date.month
        # Walk forward month by month, returning the first anchor day on/after after_date.
        for _ in range(1200):  # ~100 years, hard safety bound
            last_day = calendar.monthrange(year, month)[1]
            for day in days:
                candidate = datetime(year, month, min(day, last_day))
                if candidate >= after_date:
                    return candidate
            month += 1
            if month > 12:
                month = 1
                year += 1
        return after_date

    def __repr__(self):
        return f"RecurringPattern({self.category}, every {self.period_days}d, ~{self.typical_amount})"


class RecurrenceDetector:
    """Detect recurring patterns from historical events."""

    def __init__(self, min_occurrences: int = 3, interval_tolerance: int = 2):
        self.min_occurrences = min_occurrences
        self.interval_tolerance = interval_tolerance

    def _salary_anchor_day(self, salaries: List[Dict]) -> int:
        """
        Return the day-of-month anchor for a salary pattern.

        FIX-B: when the most recent salary lands on a different day of month
        than the modal historical payday, re-anchor to the most recent payday —
        but only if it belongs to the same description stream as the modal-day
        events. A different income stream (commission, gig, arrears, etc.)
        must never re-anchor the pattern.
        """
        salaries = sorted(salaries, key=lambda e: e['settlement_date'])
        days = [e['settlement_date'].day for e in salaries]
        modal = max(set(days), key=days.count)
        last = salaries[-1]
        last_day = last['settlement_date'].day
        if modal != last_day:
            modal_descs = {e['description'] for e, d in zip(salaries, days) if d == modal}
            if last['description'] in modal_descs:
                return last_day
        return modal

    def detect_patterns(self, events: List[Dict], scheduled_events: List[Dict] = None) -> List[RecurringPattern]:
        """
        Detect recurring patterns from events.

        Args:
            events: List of event dictionaries (must be settled events before request_date)
            scheduled_events: Optional list of scheduled future events to help seed patterns

        Returns:
            List of detected recurring patterns
        """
        # Filter to settled events only
        settled_events = [e for e in events if e['status'] == 'settled']

        # For pattern detection, exclude salary events that indicate employment has ended
        # (but keep them in settled_events for historical balance)
        def _is_terminated_salary(event):
            if event['category'] != 'salary':
                return False
            desc_lower = event['description'].lower()
            return any(term in desc_lower for term in [
                'final employer payroll', 'final payroll', 'final salary',
                'last payroll', 'employment ended'
            ])

        # Create a copy for pattern detection that excludes terminated salaries
        events_for_patterns = [
            e for e in settled_events
            if not (e['category'] == 'salary' and _is_terminated_salary(e))
        ]

        patterns = []
        salary_pattern_created = False

        # Special handling for salary: if we have a scheduled salary, assume it continues monthly
        if scheduled_events:
            scheduled_salaries = [e for e in scheduled_events if e['category'] == 'salary']
            historical_salaries = [e for e in events_for_patterns if e['category'] == 'salary']

            if scheduled_salaries and len(historical_salaries) >= 1:
                # We have at least one historical salary and a confirmed future one
                # Assume a monthly salary pattern anchored to the usual day of month
                all_salaries = historical_salaries + scheduled_salaries
                all_salaries.sort(key=lambda x: x['settlement_date'])

                # Use the most recent historical salary amount as typical (current level)
                if historical_salaries:
                    typical_amount = historical_salaries[-1]['amount']
                else:
                    # Fallback to scheduled salary if no historical
                    typical_amount = scheduled_salaries[0]['amount']
                last_date = all_salaries[-1]['settlement_date']

                # Anchor to the salary payday (modal day, re-anchored to the most
                # recent payday when the stream shifted — see _salary_anchor_day)
                anchor_day = self._salary_anchor_day(historical_salaries)

                salary_pattern = RecurringPattern(
                    category='salary',
                    period_days=0,  # Monthly, anchored to a day of month
                    typical_amount=typical_amount,
                    last_date=last_date,
                    event_ids=[e['event_id'] for e in all_salaries],
                    flexibility='fixed',
                    day_of_month=anchor_day
                )
                patterns.append(salary_pattern)
                salary_pattern_created = True

        # If no scheduled salary confirmed a pattern but historical salaries exist
        # (with no evidence of termination — already filtered above), assume the
        # salary continues monthly. Use the median historical amount as typical.
        if not salary_pattern_created:
            historical_salaries = [e for e in events_for_patterns if e['category'] == 'salary']
            # Exclude one-time salary adjustments (bonus, arrears, commission, etc.)
            # the same way the generic grouping below does
            historical_salaries = [
                e for e in historical_salaries
                if not any(word in e['description'].lower() for word in
                           ['bonus', 'arrears', 'commission', 'adjustment', 'backpay', 'overtime'])
            ]
            if len(historical_salaries) >= 1:
                all_salaries = sorted(historical_salaries, key=lambda x: x['settlement_date'])
                typical_amount = statistics.median(
                    [e['amount'] for e in historical_salaries if e['amount'] is not None]
                )
                last_date = all_salaries[-1]['settlement_date']

                # Anchor to the salary payday (modal day, re-anchored to the most
                # recent payday when the stream shifted — see _salary_anchor_day)
                anchor_day = self._salary_anchor_day(historical_salaries)

                salary_pattern = RecurringPattern(
                    category='salary',
                    period_days=0,  # Monthly, anchored to a day of month
                    typical_amount=typical_amount,
                    last_date=last_date,
                    event_ids=[e['event_id'] for e in all_salaries],
                    flexibility='fixed',
                    day_of_month=anchor_day
                )
                patterns.append(salary_pattern)
                salary_pattern_created = True

        # FIX-A: if the most recent settled salary event's description explicitly
        # indicates employment has ended, do not forecast a recurring salary
        # pattern at all. Only these event-level keywords suppress the pattern;
        # generic seasonal-contract messages must not.
        def _last_salary_terminated(events):
            salaries = [e for e in events if e['category'] == 'salary']
            if not salaries:
                return False
            last = max(salaries, key=lambda e: e['settlement_date'])
            desc_lower = str(last['description']).lower()
            return any(term in desc_lower for term in [
                'final', 'last payroll', 'termination', 'severance'
            ])

        if _last_salary_terminated(settled_events):
            # Remove any salary pattern that was created above
            patterns = [p for p in patterns if p.category != 'salary']

        if len(events_for_patterns) < self.min_occurrences:
            return patterns

        # Group events by (category, direction, flexibility)
        groups = defaultdict(list)
        for event in events_for_patterns:
            # Skip salary if we already created a pattern for it
            if salary_pattern_created and event['category'] == 'salary':
                continue

            # Skip one-time salary adjustments (bonus, arrears, commission, etc.)
            if event['category'] == 'salary':
                desc_lower = event['description'].lower()
                if any(word in desc_lower for word in ['bonus', 'arrears', 'commission', 'adjustment', 'backpay', 'overtime']):
                    continue

            # Create a key that identifies potentially recurring events
            # Use category + direction as key (not description, as it may vary slightly)
            key = (
                event['category'],
                event['direction'],
                event.get('flexibility', 'fixed')
            )
            groups[key].append(event)

        # Continue adding to existing patterns list (don't reset it)
        for key, group_events in groups.items():
            category, direction, flexibility = key

            # Check if we have scheduled events that can seed this pattern (especially for income)
            combined_for_detection = group_events[:]
            if len(group_events) < self.min_occurrences and scheduled_events:
                # Look for scheduled events of same category
                matching_scheduled = [e for e in scheduled_events
                                     if e['category'] == category and e['direction'] == direction]
                if matching_scheduled:
                    # Combine settled and scheduled to detect pattern
                    combined_for_detection = group_events + matching_scheduled

            if len(combined_for_detection) < self.min_occurrences:
                continue

            # Sort by settlement date
            combined_for_detection.sort(key=lambda x: x['settlement_date'])

            # Try to detect periodic pattern
            pattern = self._detect_periodic_pattern(combined_for_detection, category, flexibility)
            if pattern:
                patterns.append(pattern)

        return patterns

    def _detect_periodic_pattern(self, events: List[Dict], category: str,
                                  flexibility: str) -> Optional[RecurringPattern]:
        """
        Detect if events follow a periodic pattern.

        Checks for common periods: 7, 14, 28, 30, 31 days.
        Also checks for monthly patterns by day of month.
        """
        if len(events) < self.min_occurrences:
            return None

        # Calculate intervals between consecutive events
        intervals = []
        for i in range(len(events) - 1):
            date1 = events[i]['settlement_date']
            date2 = events[i + 1]['settlement_date']
            interval_days = (date2 - date1).days
            intervals.append(interval_days)

        if not intervals:
            return None

        # Check for common recurring periods
        common_periods = [7, 10, 13, 14, 15, 21, 28, 29, 30, 31]  # weekly, ~10-day, semi-monthly, bi-weekly, ~3-week, monthly

        for period in common_periods:
            matching_intervals = [
                interval for interval in intervals
                if abs(interval - period) <= self.interval_tolerance
            ]

            # If most intervals match this period, consider it recurring
            if len(matching_intervals) >= len(intervals) * 0.7:  # 70% threshold
                # Default: treat as a fixed-length period
                period_days = period
                day_of_month = None

                # If the period is approximately semi-monthly, check for consistent two anchor days per month
                if period in [13, 14, 15]:  # ~14 days
                    # Check if events consistently fall on two specific days of month
                    if len(events) >= 4:
                        days = [e['settlement_date'].day for e in events]
                        distinct_days = sorted(set(days))
                        # If we see exactly two distinct days occurring with roughly equal frequency,
                        # it's likely a semi-monthly pattern on those two days
                        if len(distinct_days) == 2:
                            day1, day2 = distinct_days
                            count1 = days.count(day1)
                            count2 = days.count(day2)
                            # If the split is reasonably balanced (e.g., 40-60 or better)
                            if min(count1, count2) >= len(days) * 0.4:
                                # Treat as semi-monthly pattern on these two days
                                period_days = 0
                                day_of_month = [day1, day2]
                # If the period is approximately monthly, check for a dominant day of month
                elif period in [28, 29, 30, 31]:
                    # Use the most common day of month so a single off-cycle event
                    # doesn't defeat an otherwise clear monthly pattern
                    month_days = [e['settlement_date'].day for e in events]
                    mode_day = max(set(month_days), key=month_days.count)
                    if month_days.count(mode_day) >= len(month_days) * 0.6:
                        # Treat as monthly pattern on the dominant day of month
                        period_days = 0
                        day_of_month = mode_day

                # Calculate typical amount (use conservative estimate for expenses)
                amounts = [e['amount'] for e in events if e['amount'] is not None]

                if not amounts:
                    return None

                # For expenses (debits), use median typical amount
                # For income (credits), use conservative estimate (10th percentile or min)
                if events[0]['direction'] == 'debit':
                    typical_amount = statistics.median(amounts)
                else:
                    typical_amount = statistics.quantiles(amounts, n=10)[0] if len(amounts) >= 10 else min(amounts)

                # Get minimum allowed amount if reducible
                minimum_allowed = events[-1].get('minimum_allowed_amount')

                return RecurringPattern(
                    category=category,
                    period_days=period_days,
                    typical_amount=typical_amount,
                    last_date=events[-1]['settlement_date'],
                    event_ids=[e['event_id'] for e in events],
                    flexibility=flexibility,
                    minimum_allowed_amount=minimum_allowed,
                    day_of_month=day_of_month
                )

        return None

    def generate_future_occurrences(self, pattern: RecurringPattern,
                                     start_date: datetime, days: int) -> List[Dict]:
        """
        Generate future occurrences of a recurring pattern.

        Args:
            pattern: Recurring pattern
            start_date: Start date for forecast
            days: Number of days to forecast

        Returns:
            List of forecasted event dictionaries
        """
        end_date = start_date + timedelta(days=days)
        occurrences = []

        # Determine direction based on category
        # Income categories are credits, expenses are debits
        if pattern.category in ['salary', 'bonus', 'commission', 'freelance']:
            direction = 'credit'
        else:
            direction = 'debit'

        if pattern.period_days == 0:
            # Monthly/semi-monthly pattern: given anchor day(s) of month
            import calendar

            days = pattern.days_of_month or ([pattern.day_of_month] if pattern.day_of_month else [1])
            year, month = start_date.year, start_date.month
            occurrence_num = 0

            # Helper to get a valid date for given year/month/day, clamping to month length
            def clamp_date(y, m, d):
                last_day = calendar.monthrange(y, m)[1]
                return datetime(y, m, min(d, last_day))

            # Find first occurrence on or after start_date
            for _ in range(24):  # scan up to 2 years ahead
                for day in days:
                    candidate = clamp_date(year, month, day)
                    if candidate >= start_date:
                        current_date = candidate
                        break
                else:
                    month += 1
                    if month > 12:
                        month = 1
                        year += 1
                    continue
                break
            else:
                return occurrences  # give up search, return empty

            # Generate all occurrences from current_date forward
            # We'll iterate by month, and within each month by anchor day
            while True:
                # Get last day of current month
                last_day = calendar.monthrange(year, month)[1]
                for day in days:
                    candidate = datetime(year, month, min(day, last_day))
                    if candidate > end_date:
                        break  # done with all months
                    if candidate >= start_date:
                        occurrences.append({
                            'event_id': f'forecast_{pattern.category}_{occurrence_num}',
                            'category': pattern.category,
                            'amount': pattern.typical_amount,
                            'settlement_date': candidate,
                            'direction': direction,
                            'status': 'forecasted',
                            'flexibility': pattern.flexibility,
                            'minimum_allowed_amount': pattern.minimum_allowed_amount,
                            'source_pattern': pattern,
                        })
                        occurrence_num += 1
                # Advance to next month
                month += 1
                if month > 12:
                    month = 1
                    year += 1
                # Check if we've gone beyond end_date
                if datetime(year, month, 1) > end_date:
                    break
        else:
            # Fixed period pattern
            current_date = pattern.next_occurrence(start_date)
            occurrence_num = 0

            while current_date <= end_date:
                occurrences.append({
                    'event_id': f'forecast_{pattern.category}_{occurrence_num}',
                    'category': pattern.category,
                    'amount': pattern.typical_amount,
                    'settlement_date': current_date,
                    'direction': direction,
                    'status': 'forecasted',
                    'flexibility': pattern.flexibility,
                    'minimum_allowed_amount': pattern.minimum_allowed_amount,
                    'source_pattern': pattern,
                })

                current_date = current_date + timedelta(days=pattern.period_days)
                occurrence_num += 1

        return occurrences

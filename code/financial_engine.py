"""
Financial engine for Buy or Wait? challenge.

Handles financial state reconstruction, forecasting, and safety checks.
"""
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import pandas as pd

from recurrence import RecurrenceDetector, RecurringPattern
from data_loader import convert_to_home_currency


class FinancialState:
    """Represents a user's financial state at a point in time."""

    def __init__(self, user_id: str, profile: Dict, as_of_date: datetime):
        self.user_id = user_id
        self.as_of_date = as_of_date
        self.home_currency = profile['home_currency']
        self.current_available_balance = profile['current_available_balance']
        self.minimum_balance_to_keep = profile['minimum_balance_to_keep']

        # Preferences
        self.financial_priorities = profile['financial_priorities']
        self.expense_categories_to_protect = profile['expense_categories_to_protect']
        self.expense_categories_user_is_willing_to_reduce = profile['expense_categories_user_is_willing_to_reduce']
        self.expense_categories_user_is_willing_to_stop = profile['expense_categories_user_is_willing_to_stop']
        self.payment_methods_user_will_consider = profile['payment_methods_user_will_consider']
        self.max_installment_months = profile['max_installment_months']

        # Events
        self.historical_events = []  # settled events before as_of_date
        self.pending_debits = []  # pending debits to reserve
        self.scheduled_events = []  # scheduled future events
        self.recurring_patterns = []  # detected recurring patterns

    def __repr__(self):
        return f"FinancialState({self.user_id}, {self.as_of_date.date()}, balance={self.current_available_balance})"


class FinancialEngine:
    """Reconstruct financial state and forecast balance."""

    def __init__(self, data: Dict, lookups: Dict, image_extractor=None):
        self.data = data
        self.lookups = lookups
        self.image_extractor = image_extractor
        self.recurrence_detector = RecurrenceDetector(min_occurrences=3, interval_tolerance=2)

    def reconstruct_state(self, user_id: str, request_date: datetime,
                          request_id: str = None) -> FinancialState:
        """
        Reconstruct financial state for a user at request date.

        Args:
            user_id: User identifier
            request_date: Date to reconstruct state
            request_id: Request identifier (for message lookup)

        Returns:
            FinancialState object
        """
        profile = self.lookups['profile_by_user'][user_id]
        state = FinancialState(user_id, profile, request_date)

        # Get user events
        user_events = self.lookups['events_by_user'].get(user_id, [])

        # Apply message updates
        messages = self.lookups['messages_by_user'].get(user_id, [])
        if request_id:
            request_messages = self.lookups['messages_by_request'].get(request_id, [])
            messages = messages + request_messages

        updated_events = self._apply_message_updates(user_events, messages, request_date)

        # Fill blank amounts from images
        updated_events = self._fill_blank_amounts(updated_events)

        # Convert foreign currency amounts to home currency
        updated_events = self._convert_to_home_currency(updated_events, state.home_currency, request_date)

        # Categorize events by status and timing
        for event in updated_events:
            event_date = event['settlement_date']

            if event['status'] in ['cancelled', 'failed']:
                # Ignore cancelled and failed events
                continue

            elif event['status'] == 'unrealized':
                # Ignore unrealized investment valuations (not cash)
                continue

            elif event['status'] == 'settled' and event_date < request_date:
                # Historical settled events
                state.historical_events.append(event)

            elif event['status'] == 'scheduled' and event_date >= request_date:
                # Scheduled future events (count as confirmed)
                state.scheduled_events.append(event)

            elif event['status'] == 'pending' and event['direction'] == 'debit':
                # Reserve pending debits
                state.pending_debits.append(event)

            elif event['status'] == 'pending' and event['direction'] == 'credit':
                # Do NOT count pending credits
                continue

        # Detect recurring patterns from historical events, using scheduled events to seed patterns
        state.recurring_patterns = self.recurrence_detector.detect_patterns(
            state.historical_events,
            state.scheduled_events
        )

        return state

    def _apply_message_updates(self, events: List[Dict], messages: List[Dict],
                               request_date: datetime) -> List[Dict]:
        """
        Apply message updates to events.

        Messages can clarify, amend, cancel, or delay events.
        """
        # Create a working copy
        events = [e.copy() for e in events]

        # Process messages in chronological order
        messages = sorted(messages, key=lambda m: m['sent_at'])

        for msg in messages:
            msg_text = msg['message_text'].lower()
            msg_date = msg['sent_at']

            # Only apply messages sent before or at request_date
            if msg_date > request_date:
                continue

            # Check for specific patterns
            if 'salary' in msg_text or 'payroll' in msg_text:
                self._handle_salary_message(events, msg, msg_text)

            elif 'refund' in msg_text and 'initiated' in msg_text and 'not' in msg_text:
                # Refund initiated but not credited - keep as pending
                self._handle_refund_pending_message(events, msg)

            elif 'pending' in msg_text and ('bonus' in msg_text or 'commission' in msg_text):
                # Pending income - do not count
                self._handle_pending_income_message(events, msg)

            elif 'contract' in msg_text and 'ended' in msg_text:
                # Contract ended - cancel future scheduled income
                self._handle_contract_end_message(events, msg, request_date)

        return events

    def _handle_salary_message(self, events: List[Dict], msg: Dict, msg_text: str):
        """Handle salary-related messages."""
        # Look for salary amount changes
        import re

        # Extract amount patterns (e.g., "EUR 1422.85", "IDR 42750000")
        amount_patterns = re.findall(r'(?:IDR|EUR|USD|INR|ZAR)\s*([0-9,]+(?:\.[0-9]+)?)', msg['message_text'])

        if amount_patterns and 'confirmed' in msg_text:
            # Update scheduled salary
            new_amount = float(amount_patterns[0].replace(',', ''))

            # Find scheduled salary events for this user
            for event in events:
                # Events already filtered by user, no need to check user_id
                if (event['category'] == 'salary' and
                    event['status'] == 'scheduled'):
                    event['amount'] = new_amount

        elif 'reduced' in msg_text and amount_patterns:
            # Reduced salary
            new_amount = float(amount_patterns[0].replace(',', ''))
            for event in events:
                # Events already filtered by user, no need to check user_id
                if (event['category'] == 'salary' and
                    event['status'] == 'scheduled'):
                    event['amount'] = new_amount

    def _handle_refund_pending_message(self, events: List[Dict], msg: Dict):
        """Handle refund pending messages - don't count pending credits."""
        if msg.get('related_event_id'):
            for event in events:
                if event['event_id'] == msg['related_event_id']:
                    # Ensure refund stays pending
                    if event['status'] == 'pending' and event['direction'] == 'credit':
                        pass  # Already handled by not counting pending credits

    def _handle_pending_income_message(self, events: List[Dict], msg: Dict):
        """Handle pending bonus/commission - mark as uncertain."""
        # Find related income events and mark them as unreliable
        for event in events:
            # Events already filtered by user
            if (event['status'] == 'scheduled' and
                ('bonus' in event['description'].lower() or 'commission' in event['description'].lower())):
                # Mark as cancelled to not count
                event['status'] = 'cancelled'

    def _handle_contract_end_message(self, events: List[Dict], msg: Dict, request_date: datetime):
        """Handle contract end messages - cancel future income."""
        for event in events:
            # Events already filtered by user
            if (event['category'] == 'salary' and
                event['status'] == 'scheduled' and
                event['settlement_date'] > request_date):
                # Cancel future salary after contract end
                event['status'] = 'cancelled'

    def _fill_blank_amounts(self, events: List[Dict]) -> List[Dict]:
        """Fill blank amounts from images."""
        events = [e.copy() for e in events]

        for event in events:
            if event['amount'] is None or (isinstance(event['amount'], float) and pd.isna(event['amount'])):
                # Check if there's an image for this event
                if event['event_id'] in self.lookups['images_by_event']:
                    image_info = self.lookups['images_by_event'][event['event_id']]
                    image_id = image_info['image_id']

                    if self.image_extractor:
                        amount = self.image_extractor.extract_amount(
                            image_id,
                            event['description'],
                            event['currency']
                        )
                        event['amount'] = amount
                    else:
                        raise ValueError(f"Event {event['event_id']} has blank amount but no image extractor available")
                else:
                    raise ValueError(f"Event {event['event_id']} has blank amount and no linked image")

        return events

    def _convert_to_home_currency(self, events: List[Dict], home_currency: str,
                                   request_date: datetime) -> List[Dict]:
        """Convert all foreign currency amounts to home currency."""
        events = [e.copy() for e in events]
        rates_df = self.lookups['exchange_rates']

        for event in events:
            if event['currency'] != home_currency and event['amount'] is not None:
                settlement_date = event['settlement_date']
                converted_amount = convert_to_home_currency(
                    event['amount'],
                    event['currency'],
                    home_currency,
                    settlement_date,
                    rates_df
                )
                event['original_amount'] = event['amount']
                event['original_currency'] = event['currency']
                event['amount'] = converted_amount
                event['currency'] = home_currency

        return events

    def forecast_balance(self, state: FinancialState, days: int = 90) -> Dict[datetime, float]:
        """
        Forecast daily balance for next N days.

        Returns:
            Dictionary mapping date to projected balance
        """
        forecast = {}
        current_balance = state.current_available_balance

        # Apply pending debits immediately
        for pending in state.pending_debits:
            current_balance -= pending['amount']

        start_date = state.as_of_date
        end_date = start_date + timedelta(days=days)

        # Collect all future cash flows
        cash_flows = defaultdict(float)  # date -> net cash flow

        # Add scheduled events
        for event in state.scheduled_events:
            event_date = event['settlement_date']
            if start_date <= event_date <= end_date:
                if event['direction'] == 'credit':
                    cash_flows[event_date] += event['amount']
                else:
                    cash_flows[event_date] -= event['amount']

        # Build set of scheduled events for deduplication
        scheduled_keys = {
            (event['settlement_date'], event['category'], event['direction'])
            for event in state.scheduled_events
        }

        # Add forecasted recurring events
        for pattern in state.recurring_patterns:
            occurrences = self.recurrence_detector.generate_future_occurrences(
                pattern, start_date, days
            )
            for occurrence in occurrences:
                event_date = occurrence['settlement_date']
                # Skip if this occurrence duplicates a scheduled event
                if (event_date, occurrence['category'], occurrence['direction']) in scheduled_keys:
                    continue
                if occurrence['direction'] == 'credit':
                    cash_flows[event_date] += occurrence['amount']
                else:
                    cash_flows[event_date] -= occurrence['amount']

        # Build daily forecast
        current_date = start_date
        while current_date <= end_date:
            # Apply cash flows for this date
            if current_date in cash_flows:
                current_balance += cash_flows[current_date]

            forecast[current_date] = current_balance
            current_date += timedelta(days=1)

        return forecast

    def calculate_safe_amount(self, state: FinancialState, requested_amount: float,
                              forecast: Dict[datetime, float]) -> float:
        """
        Calculate the maximum safe amount to pay on request date.

        Uses binary search to find largest amount <= requested_amount that passes safety check.
        """
        if self._is_amount_safe(requested_amount, state.as_of_date, state.minimum_balance_to_keep, forecast):
            return requested_amount

        # Binary search
        low, high = 0.0, requested_amount
        precision = 0.01  # 2 decimal places

        while high - low > precision:
            mid = (low + high) / 2.0
            if self._is_amount_safe(mid, state.as_of_date, state.minimum_balance_to_keep, forecast):
                low = mid
            else:
                high = mid

        return round(low, 2)

    def _is_amount_safe(self, amount: float, payment_date: datetime,
                       minimum_balance: float, forecast: Dict[datetime, float]) -> bool:
        """Check if paying amount on payment_date keeps balance above minimum for 90 days."""
        for date, balance in forecast.items():
            adjusted_balance = balance
            if date >= payment_date:
                adjusted_balance -= amount

            if adjusted_balance < minimum_balance:
                return False

        return True

    def find_earliest_full_payment_date(self, state: FinancialState, requested_amount: float,
                                        forecast: Dict[datetime, float]) -> Optional[datetime]:
        """
        Find earliest date when full amount can be safely paid.

        Returns None if not possible within forecast period.
        """
        for date in sorted(forecast.keys()):
            if date < state.as_of_date:
                continue

            # Check if paying full amount on this date is safe
            if self._is_amount_safe(requested_amount, date, state.minimum_balance_to_keep, forecast):
                return date

        return None

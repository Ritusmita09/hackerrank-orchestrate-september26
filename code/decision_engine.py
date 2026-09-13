"""
Decision engine for Buy or Wait? challenge.

Generates and selects optimal payment plans.
"""
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PaymentPlan:
    """Represents a payment plan."""
    method: str  # full_payment, partial_payment, installments, wait, not_recommended
    payments: List[Tuple[datetime, float]]  # [(date, amount), ...]
    total_payable: float
    affordability_status: str
    spending_changes: List[str]  # ["stop:event_id", "reduce_to:event_id:amount"]
    earliest_full_payment_date: Optional[datetime]
    payment_option_id: Optional[str] = None  # For installments
    explanation: str = ""

    def is_valid(self) -> bool:
        """Check if plan is internally consistent."""
        if self.method == 'not_recommended':
            return len(self.payments) == 0
        if self.method == 'wait':
            return len(self.payments) == 1
        if self.method == 'full_payment':
            return len(self.payments) == 1
        if self.method == 'partial_payment':
            return len(self.payments) == 2
        if self.method == 'installments':
            return len(self.payments) >= 2
        return False


class DecisionEngine:
    """Generate and select optimal payment plans."""

    def __init__(self, financial_engine):
        self.financial_engine = financial_engine

    def generate_recommendation(self, request: Dict, state, payment_options: List[Dict],
                               forecast: Dict[datetime, float], amount_safe_to_pay: float,
                               earliest_full_payment_date: Optional[datetime]) -> PaymentPlan:
        """
        Generate optimal payment recommendation.

        Args:
            request: Request dictionary
            state: FinancialState object
            payment_options: List of available payment options
            forecast: Balance forecast
            amount_safe_to_pay: Maximum safe amount today (before spending changes)
            earliest_full_payment_date: Earliest date for full payment (without spending changes)

        Returns:
            PaymentPlan object
        """
        request_date = request['request_date']
        requested_amount = request['requested_amount']
        desired_completion_date = request['desired_completion_date']
        allows_partial = request['allows_partial_payment']

        # Generate all possible plans
        candidate_plans = []

        # 1. Try full payment now
        if amount_safe_to_pay >= requested_amount and 'full_payment' in state.payment_methods_user_will_consider:
            plan = PaymentPlan(
                method='full_payment',
                payments=[(request_date, requested_amount)],
                total_payable=requested_amount,
                affordability_status='affordable_now',
                spending_changes=[],
                earliest_full_payment_date=request_date,
                explanation=self._explain_full_payment_now(requested_amount, state)
            )
            candidate_plans.append(plan)

        # 2. Try installments
        if 'installments' in state.payment_methods_user_will_consider:
            installment_plans = self._generate_installment_plans(
                request, state, payment_options, forecast
            )
            candidate_plans.extend(installment_plans)

        # 3. Try partial payment
        if (allows_partial and 'partial_payment' in state.payment_methods_user_will_consider and
            0 < amount_safe_to_pay < requested_amount and
            earliest_full_payment_date and earliest_full_payment_date <= desired_completion_date):

            remainder = requested_amount - amount_safe_to_pay
            plan = PaymentPlan(
                method='partial_payment',
                payments=[(request_date, amount_safe_to_pay),
                         (earliest_full_payment_date, remainder)],
                total_payable=requested_amount,
                affordability_status='affordable_with_plan',
                spending_changes=[],
                earliest_full_payment_date=earliest_full_payment_date,
                explanation=self._explain_partial_payment(amount_safe_to_pay, remainder,
                                                         earliest_full_payment_date, state)
            )

            # Validate this plan
            if self._validate_plan(plan, state, forecast, request):
                candidate_plans.append(plan)

        # 4. Try full payment with spending changes
        if amount_safe_to_pay < requested_amount:
            spending_changes = self._generate_spending_changes(
                state, requested_amount, forecast
            )

            if spending_changes:
                # Apply spending changes and recalculate forecast
                modified_forecast = self._apply_spending_changes_to_forecast(
                    forecast, spending_changes, state
                )

                # Check if now affordable
                if self.financial_engine._is_amount_safe(
                    requested_amount, request_date, state.minimum_balance_to_keep, modified_forecast
                ):
                    if 'full_payment' in state.payment_methods_user_will_consider:
                        plan = PaymentPlan(
                            method='full_payment',
                            payments=[(request_date, requested_amount)],
                            total_payable=requested_amount,
                            affordability_status='affordable_with_plan',
                            spending_changes=spending_changes,
                            earliest_full_payment_date=earliest_full_payment_date,
                            explanation=self._explain_with_changes(requested_amount, spending_changes, state)
                        )
                        candidate_plans.append(plan)

        # 5. Try wait (if full payment becomes safe later)
        if (earliest_full_payment_date and
            earliest_full_payment_date > request_date and
            'full_payment' in state.payment_methods_user_will_consider):

            plan = PaymentPlan(
                method='wait',
                payments=[(earliest_full_payment_date, requested_amount)],
                total_payable=requested_amount,
                affordability_status='affordable_later',
                spending_changes=[],
                earliest_full_payment_date=earliest_full_payment_date,
                explanation=self._explain_wait(requested_amount, earliest_full_payment_date, state)
            )
            candidate_plans.append(plan)

        # 6. Select best plan
        if candidate_plans:
            best_plan = self._rank_and_select(candidate_plans, desired_completion_date)
            return best_plan

        # 7. Fallback: not recommended
        return PaymentPlan(
            method='not_recommended',
            payments=[],
            total_payable=0,
            affordability_status='not_affordable',
            spending_changes=[],
            earliest_full_payment_date=earliest_full_payment_date,
            explanation=self._explain_not_recommended(amount_safe_to_pay, requested_amount, state)
        )

    def _generate_installment_plans(self, request: Dict, state, payment_options: List[Dict],
                                    forecast: Dict[datetime, float]) -> List[PaymentPlan]:
        """Generate installment plans from available payment options."""
        plans = []
        request_date = request['request_date']
        requested_amount = request['requested_amount']

        for option in payment_options:
            if option['payment_method'] != 'installments':
                continue

            # Check user's max installment months
            if pd.notna(state.max_installment_months) and state.max_installment_months:
                if option['number_of_payments'] > state.max_installment_months:
                    continue

            # Build payment schedule
            payments = []
            current_date = option['first_payment_date']
            payment_amount = option['payment_amount']
            num_payments = option['number_of_payments']
            frequency_days = option['payment_frequency_days']

            for i in range(num_payments):
                payments.append((current_date, payment_amount))
                if i < num_payments - 1:
                    current_date = current_date + timedelta(days=frequency_days)

            # Create plan
            plan = PaymentPlan(
                method='installments',
                payments=payments,
                total_payable=option['total_payable_amount'],
                affordability_status='affordable_with_plan',
                spending_changes=[],
                earliest_full_payment_date=None,  # Will be set separately
                payment_option_id=option['payment_option_id'],
                explanation=self._explain_installments(num_payments, option['first_payment_date'], state)
            )

            # Validate this plan
            if self._validate_plan(plan, state, forecast, request):
                plans.append(plan)

        return plans

    def _validate_plan(self, plan: PaymentPlan, state, forecast: Dict[datetime, float],
                      request: Dict) -> bool:
        """
        Validate that a plan passes the 90-day safety check.

        For each payment in the plan, check that balance stays above minimum.
        """
        if not plan.payments:
            return True

        # Apply spending changes if any
        working_forecast = forecast.copy()
        if plan.spending_changes:
            working_forecast = self._apply_spending_changes_to_forecast(
                working_forecast, plan.spending_changes, state
            )

        # Check each payment
        for payment_date, payment_amount in plan.payments:
            # Check if this payment is safe
            if not self.financial_engine._is_amount_safe(
                payment_amount, payment_date, state.minimum_balance_to_keep, working_forecast
            ):
                return False

            # Apply this payment to forecast for subsequent payments
            working_forecast = {
                date: (balance - payment_amount if date == payment_date else balance)
                for date, balance in working_forecast.items()
            }

        # Check completion date
        if plan.payments[-1][0] > request['desired_completion_date']:
            return False

        return True

    def _generate_spending_changes(self, state, requested_amount: float,
                                   forecast: Dict[datetime, float]) -> List[str]:
        """
        Search for up to 3 permitted spending changes that make paying
        requested_amount on the request date pass the 90-day safety check.

        Candidates are verified against the actual modified forecast floor
        (not a cumulative-savings heuristic). One action per pattern:
        a reducible pattern is reduced when the user will reduce that
        category, else stopped when they will stop it. The change is
        anchored on the pattern's most recent event. Fewer changes and
        less total relief are preferred (minimal disruption).
        """
        import itertools

        payment_date = state.as_of_date

        candidates = []
        for pattern in state.recurring_patterns:
            flex = pattern.flexibility
            cat = pattern.category
            reduce_ok = (pattern.minimum_allowed_amount is not None and
                         pd.notna(pattern.minimum_allowed_amount) and
                         ('reducible' in flex) and
                         cat in state.expense_categories_user_is_willing_to_reduce)
            stop_ok = (('stoppable' in flex) and
                       cat in state.expense_categories_user_is_willing_to_stop)
            if reduce_ok:
                candidates.append(('reduce', pattern))
            elif stop_ok:
                candidates.append(('stop', pattern))

        if not candidates:
            return []

        def per_occurrence_relief(cand):
            action, pattern = cand
            if action == 'stop':
                return pattern.typical_amount
            return pattern.typical_amount - pattern.minimum_allowed_amount

        def to_change(cand):
            action, pattern = cand
            event_id = pattern.event_ids[-1] if pattern.event_ids else f"pattern_{pattern.category}"
            if action == 'stop':
                return f"stop:{event_id}"
            return f"reduce_to:{event_id}:{pattern.minimum_allowed_amount}"

        def total_relief(subset):
            return sum(per_occurrence_relief(c) for c in subset)

        # Enumerate subsets of increasing size; within a size prefer less
        # total relief (minimal disruption). Return the first subset whose
        # changes make the full payment pass the actual safety check.
        for size in (1, 2, 3):
            subsets = list(itertools.combinations(range(len(candidates)), size))
            subsets.sort(key=lambda s: total_relief([candidates[i] for i in s]))
            for subset_idx in subsets:
                subset = [candidates[i] for i in subset_idx]
                changes = [to_change(c) for c in subset]
                modified = self._apply_spending_changes_to_forecast(forecast, changes, state)
                if self.financial_engine._is_amount_safe(
                    requested_amount, payment_date, state.minimum_balance_to_keep, modified
                ):
                    return changes

        return []

    def _apply_spending_changes_to_forecast(self, forecast: Dict[datetime, float],
                                           changes: List[str], state) -> Dict[datetime, float]:
        """
        Apply spending changes to forecast.

        Returns modified forecast.
        """
        modified = forecast.copy()

        for change in changes:
            if change.startswith('stop:'):
                event_id = change.split(':')[1]
                # Find pattern and remove its future occurrences
                for pattern in state.recurring_patterns:
                    if event_id in pattern.event_ids:
                        # Recalculate forecast without this pattern
                        occurrences = self.financial_engine.recurrence_detector.generate_future_occurrences(
                            pattern, state.as_of_date, 90
                        )
                        for occ in occurrences:
                            occ_date = occ['settlement_date']
                            if occ_date in modified:
                                modified[occ_date] += pattern.typical_amount  # Add back (remove debit)

            elif change.startswith('reduce_to:'):
                parts = change.split(':')
                event_id = parts[1]
                new_amount = float(parts[2])

                # Find pattern and reduce its future occurrences
                for pattern in state.recurring_patterns:
                    if event_id in pattern.event_ids:
                        reduction = pattern.typical_amount - new_amount
                        occurrences = self.financial_engine.recurrence_detector.generate_future_occurrences(
                            pattern, state.as_of_date, 90
                        )
                        for occ in occurrences:
                            occ_date = occ['settlement_date']
                            if occ_date in modified:
                                modified[occ_date] += reduction  # Add back the reduction

        return modified

    def _rank_and_select(self, plans: List[PaymentPlan], desired_completion_date: datetime) -> PaymentPlan:
        """
        Rank plans and select the best one.

        Ranking criteria (in order):
        1. Completes by desired_completion_date
        2. No spending changes
        3. Lower total payable amount
        4. Earlier start date
        5. Fewer payments
        6. Lower payment_option_id (for tie-breaking installments)
        """
        def rank_key(plan: PaymentPlan):
            completes_on_time = plan.payments[-1][0] <= desired_completion_date if plan.payments else False
            has_no_changes = len(plan.spending_changes) == 0
            total_payable = plan.total_payable
            start_date = plan.payments[0][0] if plan.payments else datetime.max
            num_payments = len(plan.payments)
            option_id = int(plan.payment_option_id.split('_')[2]) if plan.payment_option_id else 999999

            return (
                not completes_on_time,  # False (completes on time) sorts first
                not has_no_changes,     # False (no changes) sorts first
                total_payable,          # Lower is better
                -start_date.timestamp(), # Earlier is better (negative for ascending)
                num_payments,           # Fewer is better
                option_id               # Lower is better
            )

        plans.sort(key=rank_key)
        return plans[0]

    def _explain_full_payment_now(self, amount: float, state) -> str:
        """Generate explanation for full payment now."""
        return (f"Pay {state.home_currency} {amount:,.2f} today. "
                f"This leaves at least {state.home_currency} {state.minimum_balance_to_keep:,.2f} "
                f"available over the next 90 days.")

    def _explain_partial_payment(self, first_amount: float, second_amount: float,
                                second_date: datetime, state) -> str:
        """Generate explanation for partial payment."""
        return (f"Pay {state.home_currency} {first_amount:,.2f} today and the remaining "
                f"{state.home_currency} {second_amount:,.2f} on {second_date.strftime('%d %B %Y')}. "
                f"This completes the full request and keeps the {state.home_currency} "
                f"{state.minimum_balance_to_keep:,.2f} minimum protected.")

    def _explain_installments(self, num_payments: int, start_date: datetime, state) -> str:
        """Generate explanation for installments."""
        return (f"Use {num_payments} installments, starting {start_date.strftime('%d %B %Y')}. "
                f"This leaves at least {state.home_currency} {state.minimum_balance_to_keep:,.2f} available.")

    def _explain_with_changes(self, amount: float, changes: List[str], state) -> str:
        """Generate explanation with spending changes."""
        change_desc = self._format_spending_changes(changes)
        return (f"{change_desc}, then pay {state.home_currency} {amount:,.2f} today. "
                f"This leaves at least {state.home_currency} {state.minimum_balance_to_keep:,.2f} available.")

    def _explain_wait(self, amount: float, wait_until: datetime, state) -> str:
        """Generate explanation for wait."""
        return (f"Pay {state.home_currency} {amount:,.2f} in full on {wait_until.strftime('%d %B %Y')}. "
                f"Paying earlier would take the balance below the {state.home_currency} "
                f"{state.minimum_balance_to_keep:,.2f} minimum.")

    def _explain_not_recommended(self, safe_amount: float, requested_amount: float, state) -> str:
        """Generate explanation for not recommended."""
        if safe_amount > 0:
            return (f"Do not proceed with the {state.home_currency} {requested_amount:,.2f} request. "
                    f"Although {state.home_currency} {safe_amount:,.2f} is available today, the full amount "
                    f"cannot be completed safely within 90 days.")
        else:
            return (f"Do not make this payment. None of the available options keeps the "
                    f"{state.home_currency} {state.minimum_balance_to_keep:,.2f} minimum protected.")

    def _format_spending_changes(self, changes: List[str]) -> str:
        """Format spending changes for explanation."""
        formatted = []
        for change in changes:
            if change.startswith('stop:'):
                formatted.append("Stop the subscription")
            elif change.startswith('reduce_to:'):
                parts = change.split(':')
                new_amount = parts[2]
                formatted.append(f"Reduce spending to {new_amount}")

        if len(formatted) == 1:
            return formatted[0]
        elif len(formatted) == 2:
            return f"{formatted[0]} and {formatted[1]}"
        else:
            return ", ".join(formatted[:-1]) + f", and {formatted[-1]}"


import pandas as pd

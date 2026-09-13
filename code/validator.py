"""
Output validator for Buy or Wait? challenge.

Validates output.csv against problem constraints.
"""
import pandas as pd
from typing import Dict, List, Tuple


class OutputValidator:
    """Validate output CSV against problem specification."""

    REQUIRED_COLUMNS = [
        'request_id',
        'amount_safe_to_pay',
        'affordability_status',
        'recommended_payment_method',
        'payment_plan',
        'earliest_date_for_full_payment',
        'spending_changes_needed',
        'decision_explanation'
    ]

    VALID_AFFORDABILITY_STATUS = {
        'affordable_now',
        'affordable_with_plan',
        'affordable_later',
        'not_affordable'
    }

    VALID_PAYMENT_METHODS = {
        'full_payment',
        'partial_payment',
        'installments',
        'wait',
        'not_recommended'
    }

    def __init__(self):
        self.errors = []
        self.warnings = []

    def validate(self, output_df: pd.DataFrame, requests_df: pd.DataFrame,
                 payment_options_df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
        """
        Validate output DataFrame.

        Returns:
            (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []

        # Check schema
        self._validate_schema(output_df)

        # Check row count
        self._validate_row_count(output_df, requests_df)

        # Check request_ids match
        self._validate_request_ids(output_df, requests_df)

        # Check each row
        for idx, row in output_df.iterrows():
            self._validate_row(row, idx, requests_df, payment_options_df)

        return len(self.errors) == 0, self.errors, self.warnings

    def _validate_schema(self, df: pd.DataFrame):
        """Validate column schema."""
        if list(df.columns) != self.REQUIRED_COLUMNS:
            self.errors.append(
                f"Column mismatch. Expected: {self.REQUIRED_COLUMNS}, Got: {list(df.columns)}"
            )

    def _validate_row_count(self, output_df: pd.DataFrame, requests_df: pd.DataFrame):
        """Validate row count."""
        expected = len(requests_df)
        actual = len(output_df)
        if actual != expected:
            self.errors.append(f"Row count mismatch. Expected {expected}, got {actual}")

    def _validate_request_ids(self, output_df: pd.DataFrame, requests_df: pd.DataFrame):
        """Validate all request_ids are present."""
        expected_ids = set(requests_df['request_id'])
        actual_ids = set(output_df['request_id'])

        missing = expected_ids - actual_ids
        extra = actual_ids - expected_ids

        if missing:
            self.errors.append(f"Missing request_ids: {missing}")
        if extra:
            self.errors.append(f"Extra request_ids: {extra}")

    def _validate_row(self, row: pd.Series, idx: int, requests_df: pd.DataFrame,
                     payment_options_df: pd.DataFrame):
        """Validate a single row."""
        request_id = row['request_id']

        # Get corresponding request
        request = requests_df[requests_df['request_id'] == request_id]
        if len(request) == 0:
            return  # Already caught by request_ids check

        request = request.iloc[0]
        requested_amount = request['requested_amount']

        # Validate amount_safe_to_pay
        amount_safe = row['amount_safe_to_pay']
        if pd.isna(amount_safe):
            self.errors.append(f"{request_id}: amount_safe_to_pay is missing")
        elif not (0 <= amount_safe <= requested_amount):
            self.errors.append(
                f"{request_id}: amount_safe_to_pay ({amount_safe}) not in [0, {requested_amount}]"
            )

        # Validate affordability_status
        if row['affordability_status'] not in self.VALID_AFFORDABILITY_STATUS:
            self.errors.append(
                f"{request_id}: Invalid affordability_status '{row['affordability_status']}'"
            )

        # Validate recommended_payment_method
        if row['recommended_payment_method'] not in self.VALID_PAYMENT_METHODS:
            self.errors.append(
                f"{request_id}: Invalid recommended_payment_method '{row['recommended_payment_method']}'"
            )

        # Validate payment_plan format
        self._validate_payment_plan(row, request_id, payment_options_df)

        # Validate spending_changes_needed format
        self._validate_spending_changes(row, request_id)

        # Validate consistency
        self._validate_consistency(row, request, request_id)

    def _validate_payment_plan(self, row: pd.Series, request_id: str,
                               payment_options_df: pd.DataFrame):
        """Validate payment_plan format and content."""
        plan = row['payment_plan']

        if pd.isna(plan) or plan == '' or plan == 'none':
            if row['recommended_payment_method'] != 'not_recommended':
                self.warnings.append(
                    f"{request_id}: payment_plan is empty but method is {row['recommended_payment_method']}"
                )
            return

        # Parse payment plan
        try:
            entries = plan.split('|')
            for entry in entries:
                date_str, amount_str = entry.split(':')
                # Validate date format
                pd.to_datetime(date_str)
                # Validate amount
                float(amount_str)
        except Exception as e:
            self.errors.append(f"{request_id}: Invalid payment_plan format: {e}")
            return

        # Validate partial payment has exactly 2 entries
        if row['recommended_payment_method'] == 'partial_payment':
            if len(entries) != 2:
                self.errors.append(
                    f"{request_id}: partial_payment must have exactly 2 payments, got {len(entries)}"
                )

    def _validate_spending_changes(self, row: pd.Series, request_id: str):
        """Validate spending_changes_needed format."""
        changes = row['spending_changes_needed']

        if pd.isna(changes) or changes == '' or changes == 'none':
            return

        # Parse spending changes
        try:
            change_list = changes.split('|')
            if len(change_list) > 3:
                self.errors.append(
                    f"{request_id}: Too many spending changes ({len(change_list)}), max 3"
                )

            for change in change_list:
                if change.startswith('stop:'):
                    pass  # Valid
                elif change.startswith('reduce_to:'):
                    parts = change.split(':')
                    if len(parts) != 3:
                        self.errors.append(f"{request_id}: Invalid reduce_to format: {change}")
                    else:
                        float(parts[2])  # Validate amount
                else:
                    self.errors.append(f"{request_id}: Invalid spending change format: {change}")
        except Exception as e:
            self.errors.append(f"{request_id}: Error parsing spending_changes_needed: {e}")

    def _validate_consistency(self, row: pd.Series, request: pd.Series, request_id: str):
        """Validate consistency between fields."""
        method = row['recommended_payment_method']
        status = row['affordability_status']

        # affordable_now should have earliest_date = request_date
        if status == 'affordable_now':
            if method not in ['full_payment', 'installments']:
                self.warnings.append(
                    f"{request_id}: affordable_now but method is {method}"
                )

        # not_affordable should have not_recommended
        if status == 'not_affordable':
            if method != 'not_recommended':
                self.errors.append(
                    f"{request_id}: not_affordable but method is {method}"
                )

        # not_recommended should have not_affordable
        if method == 'not_recommended':
            if status != 'not_affordable':
                self.errors.append(
                    f"{request_id}: not_recommended but status is {status}"
                )


def validate_output_file(output_path: str, dataset_dir: str = "dataset") -> bool:
    """
    Validate output.csv file.

    Returns True if valid, False otherwise.
    """
    import sys
    from pathlib import Path

    output_df = pd.read_csv(output_path)
    requests_df = pd.read_csv(Path(dataset_dir) / "requests.csv")
    payment_options_df = pd.read_csv(Path(dataset_dir) / "request_payment_options.csv")

    validator = OutputValidator()
    is_valid, errors, warnings = validator.validate(output_df, requests_df, payment_options_df)

    if warnings:
        print("\n[WARNINGS]:")
        for warning in warnings:
            print(f"  {warning}")

    if errors:
        print("\n[ERRORS]:")
        for error in errors:
            print(f"  {error}")
        print(f"\nValidation FAILED: {len(errors)} errors found")
        return False
    else:
        print("[OK] Validation PASSED")
        return True

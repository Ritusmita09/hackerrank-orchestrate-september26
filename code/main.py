"""
Main entry point for Buy or Wait? challenge.

Generates output.csv with predictions for all 250 requests.
"""
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

from data_loader import DataLoader
from image_extractor import ImageExtractor
from financial_engine import FinancialEngine
from decision_engine import DecisionEngine
from validator import OutputValidator
from usage_logger import UsageLogger


def format_payment_plan(plan) -> str:
    """Format payment plan for output CSV."""
    if not plan.payments:
        return 'none'

    entries = []
    for date, amount in plan.payments:
        # Format without decimal places if it's a whole number
        if amount == int(amount):
            entries.append(f"{date.strftime('%Y-%m-%d')}:{int(amount)}")
        else:
            entries.append(f"{date.strftime('%Y-%m-%d')}:{amount:.2f}")

    return '|'.join(entries)


def format_spending_changes(changes: list) -> str:
    """Format spending changes for output CSV."""
    if not changes:
        return 'none'
    return '|'.join(changes)


def format_earliest_date(date) -> str:
    """Format earliest date for output CSV."""
    if date is None:
        return ''
    return date.strftime('%Y-%m-%d')


def process_request(request: dict, data: dict, lookups: dict,
                   financial_engine: FinancialEngine,
                   decision_engine: DecisionEngine) -> dict:
    """
    Process a single request and return prediction.

    Returns dictionary with all output fields.
    """
    request_id = request['request_id']
    user_id = request['user_id']
    request_date = request['request_date']
    requested_amount = request['requested_amount']

    print(f"Processing {request_id} (user {user_id}, {requested_amount:,.2f})...")

    # Reconstruct financial state
    state = financial_engine.reconstruct_state(user_id, request_date, request_id)

    # Forecast balance for 90 days
    forecast = financial_engine.forecast_balance(state, days=90)

    # Calculate safe amount to pay today (before spending changes)
    amount_safe_to_pay = financial_engine.calculate_safe_amount(
        state, requested_amount, forecast
    )

    # Find earliest date for full payment (without spending changes)
    earliest_full_payment_date = financial_engine.find_earliest_full_payment_date(
        state, requested_amount, forecast
    )

    # Get payment options for this request
    payment_options = lookups['options_by_request'].get(request_id, [])

    # Generate recommendation
    plan = decision_engine.generate_recommendation(
        request, state, payment_options, forecast,
        amount_safe_to_pay, earliest_full_payment_date
    )

    # Build output record
    output = {
        'request_id': request_id,
        'amount_safe_to_pay': amount_safe_to_pay,
        'affordability_status': plan.affordability_status,
        'recommended_payment_method': plan.method,
        'payment_plan': format_payment_plan(plan),
        'earliest_date_for_full_payment': format_earliest_date(earliest_full_payment_date),
        'spending_changes_needed': format_spending_changes(plan.spending_changes),
        'decision_explanation': plan.explanation,
    }

    return output


def main():
    """Main entry point."""
    print("=" * 80)
    print("Buy or Wait? - Financial Decision Agent")
    print("=" * 80)
    print()

    # Initialize components
    print("Initializing...")
    loader = DataLoader()
    image_extractor = ImageExtractor()
    usage_logger = UsageLogger()

    # Load data
    data = loader.load_all()
    lookups = loader.build_lookup_structures(data)
    print()

    # Initialize engines
    financial_engine = FinancialEngine(data, lookups, image_extractor)
    decision_engine = DecisionEngine(financial_engine)

    # Process all requests
    print("Processing requests...")
    print()

    results = []
    requests = data['requests'].to_dict('records')

    for request in requests:
        try:
            output = process_request(request, data, lookups, financial_engine, decision_engine)
            results.append(output)
        except Exception as e:
            print(f"  ERROR processing {request['request_id']}: {e}")
            import traceback
            traceback.print_exc()
            # Add empty result to maintain row count
            results.append({
                'request_id': request['request_id'],
                'amount_safe_to_pay': 0,
                'affordability_status': 'not_affordable',
                'recommended_payment_method': 'not_recommended',
                'payment_plan': 'none',
                'earliest_date_for_full_payment': '',
                'spending_changes_needed': 'none',
                'decision_explanation': f'Error: {str(e)}',
            })

    print()
    print(f"Processed {len(results)} requests")
    print()

    # Create output DataFrame
    output_df = pd.DataFrame(results)

    # Write output.csv
    output_path = "output.csv"
    output_df.to_csv(output_path, index=False)
    print(f"[OK] Written output to {output_path}")
    print()

    # Validate output
    print("Validating output...")
    validator = OutputValidator()
    is_valid, errors, warnings = validator.validate(
        output_df, data['requests'], data['payment_options']
    )

    if warnings:
        print(f"\n[WARN] {len(warnings)} warnings")
        for warning in warnings[:5]:  # Show first 5
            print(f"  {warning}")

    if errors:
        print(f"\n[ERROR] {len(errors)} errors")
        for error in errors[:5]:  # Show first 5
            print(f"  {error}")
    else:
        print("[OK] Validation passed")

    print()

    # Generate usage report
    print("Generating usage report...")
    usage_stats = image_extractor.get_usage_stats()
    usage_logger.log_call(
        'Anthropic',
        'claude-3-5-sonnet-20241022',
        usage_stats['input_tokens'],
        usage_stats['output_tokens']
    )
    usage_logger.set_total_requests(len(requests))
    usage_logger.generate_report()

    summary = usage_logger.get_summary()
    print(f"  Total API calls: {usage_stats['calls']}")
    print(f"  Total tokens: {summary['total_tokens']:,}")
    print(f"  Estimated cost: ${summary['estimated_cost']:.4f}")
    print()

    print("=" * 80)
    print("[DONE]")
    print("=" * 80)

    return 0 if is_valid else 1


if __name__ == '__main__':
    sys.exit(main())

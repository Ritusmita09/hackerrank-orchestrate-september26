"""
Evaluation script for Buy or Wait? challenge.

Tests implementation against 25 sample requests with known answers.
"""
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import DataLoader
from image_extractor import ImageExtractor
from financial_engine import FinancialEngine
from decision_engine import DecisionEngine
from main import process_request


def compare_field(actual, expected, field_name: str, request_id: str) -> bool:
    """Compare a single field and report mismatch."""
    # Normalize for comparison
    actual_str = str(actual).strip() if pd.notna(actual) else ''
    expected_str = str(expected).strip() if pd.notna(expected) else ''

    if actual_str == expected_str:
        return True

    # For date fields, try parsing and comparing as dates
    if field_name == 'earliest_date_for_full_payment':
        try:
            if actual_str and expected_str:
                # Parse both as dates and compare
                actual_date = pd.to_datetime(actual_str).date()
                expected_date = pd.to_datetime(expected_str).date()
                if actual_date == expected_date:
                    return True
        except:
            pass

    # For numeric fields, try comparing as floats
    if field_name in ['amount_safe_to_pay', 'requested_amount']:
        try:
            actual_num = float(actual_str) if actual_str else 0
            expected_num = float(expected_str) if expected_str else 0
            if abs(actual_num - expected_num) < 0.01:  # Allow small floating point differences
                return True
        except:
            pass

    print(f"  [MISMATCH] {field_name}")
    print(f"     Expected: {expected_str}")
    print(f"     Actual:   {actual_str}")
    return False


def evaluate_sample_requests():
    """Evaluate against 25 sample requests."""
    print("=" * 80)
    print("Sample Request Evaluation")
    print("=" * 80)
    print()

    # Initialize components
    print("Initializing...")
    loader = DataLoader()
    image_extractor = ImageExtractor()

    # Load data
    data = loader.load_all()
    lookups = loader.build_lookup_structures(data)
    print()

    # Initialize engines
    financial_engine = FinancialEngine(data, lookups, image_extractor)
    decision_engine = DecisionEngine(financial_engine)

    # Load sample requests
    sample_requests = data['sample_requests'].to_dict('records')

    print(f"Evaluating {len(sample_requests)} sample requests...")
    print()

    total_matches = 0
    total_mismatches = 0
    field_accuracy = {}

    for sample in sample_requests:
        request_id = sample['request_id']
        print(f"{request_id}:")

        # Process request
        try:
            output = process_request(sample, data, lookups, financial_engine, decision_engine)

            # Compare fields
            fields_to_compare = [
                'amount_safe_to_pay',
                'affordability_status',
                'recommended_payment_method',
                'payment_plan',
                'earliest_date_for_full_payment',
                'spending_changes_needed',
            ]

            request_matches = 0
            request_mismatches = 0

            for field in fields_to_compare:
                if field not in field_accuracy:
                    field_accuracy[field] = {'matches': 0, 'mismatches': 0}

                if compare_field(output[field], sample[field], field, request_id):
                    field_accuracy[field]['matches'] += 1
                    request_matches += 1
                else:
                    field_accuracy[field]['mismatches'] += 1
                    request_mismatches += 1

            if request_mismatches == 0:
                print("  [OK] All fields match")
                total_matches += 1
            else:
                print(f"  [WARN] {request_mismatches} field(s) mismatch")
                total_mismatches += 1

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            total_mismatches += 1

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Requests with all fields matching: {total_matches}/{len(sample_requests)}")
    print(f"Requests with mismatches: {total_mismatches}/{len(sample_requests)}")
    print()

    print("Field-by-field accuracy:")
    for field, stats in sorted(field_accuracy.items()):
        total = stats['matches'] + stats['mismatches']
        accuracy = (stats['matches'] / total * 100) if total > 0 else 0
        print(f"  {field:40s}: {stats['matches']:2d}/{total:2d} ({accuracy:5.1f}%)")

    print()
    print("=" * 80)

    if total_mismatches == 0:
        print("[PASS] ALL SAMPLES PASSED")
    else:
        print(f"[FAIL] {total_mismatches} samples had mismatches - review and fix")

    print("=" * 80)

    return 0 if total_mismatches == 0 else 1


if __name__ == '__main__':
    sys.exit(evaluate_sample_requests())

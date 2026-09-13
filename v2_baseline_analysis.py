"""V2 baseline analysis: per-request field matrix + amount error structure (read-only)."""
import sys, re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "code"))

from data_loader import DataLoader
from image_extractor import ImageExtractor
from financial_engine import FinancialEngine
from decision_engine import DecisionEngine
from main import process_request

loader = DataLoader()
data = loader.load_all()
lookups = loader.build_lookup_structures(data)
fe = FinancialEngine(data, lookups, ImageExtractor())
de = DecisionEngine(fe)

FIELDS = ['amount_safe_to_pay', 'affordability_status', 'recommended_payment_method',
          'payment_plan', 'earliest_date_for_full_payment', 'spending_changes_needed']

rows = []
for sample in data['sample_requests'].to_dict('records'):
    out = process_request(sample, data, lookups, fe, de)
    row = {'request_id': sample['request_id']}
    for f in FIELDS:
        a = str(out[f]).strip()
        e = str(sample[f]).strip() if pd.notna(sample[f]) else ''
        if f == 'amount_safe_to_pay':
            ok = abs(float(a) - float(e)) < 0.01 if a and e else a == e
        elif f == 'earliest_date_for_full_payment':
            ok = (a[:10] == e[:10]) if (a or e) else (not a and not e)
        else:
            ok = a == e
        row[f + '_ok'] = ok
    req_amt = sample['requested_amount']
    exp = float(sample['amount_safe_to_pay'])
    act = float(out['amount_safe_to_pay'])
    row['requested'] = req_amt
    row['expected_amt'] = exp
    row['actual_amt'] = act
    row['abs_err'] = round(act - exp, 2)
    row['pct_of_req'] = round((act - exp) / req_amt * 100, 2) if req_amt else None
    row['expected_frac_of_req'] = round(exp / req_amt * 100, 2) if req_amt else None
    rows.append(row)

df = pd.DataFrame(rows)
print(df[['request_id'] + [f + '_ok' for f in FIELDS]].to_string(index=False))
print()
print("Amount error structure (actual - expected):")
print(df[['request_id', 'requested', 'expected_amt', 'actual_amt', 'abs_err', 'pct_of_req',
          'expected_frac_of_req']].to_string(index=False))
print()
for f in FIELDS:
    print(f"{f:40s}: {df[f + '_ok'].sum()}/25")
print(f"{'FULL ROW':40s}: {df[[f + '_ok' for f in FIELDS]].all(axis=1).sum()}/25")
labeled = df[[f + '_ok' for f in FIELDS]].values.sum()
print(f"{'OVERALL LABELED FIELDS':40s}: {labeled}/150 = {labeled/150*100:.1f}%")
print()
print("Over-optimistic (act>exp):", (df['abs_err'] > 0.01).sum(),
      "| Pessimistic (act<exp):", (df['abs_err'] < -0.01).sum(),
      "| Exact:", (df['abs_err'].abs() <= 0.01).sum())

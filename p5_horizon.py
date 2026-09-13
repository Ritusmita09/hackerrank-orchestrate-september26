"""P4 final: horizon-shape hypothesis. Test safe amount when the constraint window
ends at (a) the first forecast credit (next payday), (b) +30d, (c) +45d, vs 90d.
Diagnostic only, all in-memory."""
import sys
from pathlib import Path
from datetime import timedelta
from collections import defaultdict
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "code"))

from data_loader import DataLoader
from image_extractor import ImageExtractor
from financial_engine import FinancialEngine

loader = DataLoader()
data = loader.load_all()
lookups = loader.build_lookup_structures(data)
fe = FinancialEngine(data, lookups, ImageExtractor())
samples = {s['request_id']: s for s in data['sample_requests'].to_dict('records')}

def build_flows(state):
    flows = defaultdict(float)
    for event in state.scheduled_events:
        d = event['settlement_date']
        if state.as_of_date <= d <= state.as_of_date + timedelta(days=90):
            flows[d] += event['amount'] if event['direction'] == 'credit' else -event['amount']
    sched_keys = {(e['settlement_date'], e['category'], e['direction']) for e in state.scheduled_events}
    first_credit = None
    for pattern in state.recurring_patterns:
        for o in fe.recurrence_detector.generate_future_occurrences(pattern, state.as_of_date, 90):
            if (o['settlement_date'], o['category'], o['direction']) in sched_keys:
                continue
            flows[o['settlement_date']] += o['amount'] if o['direction'] == 'credit' else -o['amount']
    for d in sorted(flows):
        if flows[d] > 0:
            first_credit = d
            break
    return flows, first_credit

def safe_amount(state, flows, horizon_end, requested):
    def ok(amount):
        bal = state.current_available_balance - sum(e['amount'] for e in state.pending_debits)
        d = state.as_of_date
        while d <= horizon_end:
            bal += flows.get(d, 0)
            if bal - amount < state.minimum_balance_to_keep - 1e-9:
                return False
            d += timedelta(days=1)
        return True
    if ok(requested):
        return requested
    lo, hi = 0.0, requested
    while hi - lo > 0.01:
        mid = (lo + hi) / 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return round(lo, 2)

print(f"{'rid':<13}{'exp':>14}{'90d':>14}{'to_payday':>13}{'+2d':>12}  first_credit")
for rid in sorted(samples, key=lambda r: int(r.split('_')[1])):
    s = samples[rid]
    state = fe.reconstruct_state(s['user_id'], s['request_date'], rid)
    flows, fc = build_flows(state)
    req = float(s['requested_amount'])
    exp = float(s['amount_safe_to_pay'])
    a90 = safe_amount(state, flows, state.as_of_date + timedelta(days=90), req)
    if fc:
        a_pay = safe_amount(state, flows, fc, req)
        a_pay2 = safe_amount(state, flows, fc + timedelta(days=2), req)
        fc_s = str(fc.date())
    else:
        a_pay = a_pay2 = None
        fc_s = 'NONE'
    marks = []
    if abs(a90 - exp) <= 0.01: marks.append('90d=exp')
    if fc and abs(a_pay - exp) <= 0.01: marks.append('payday=exp')
    if fc and abs(a_pay2 - exp) <= 0.01: marks.append('payday+2=exp')
    def f(v): return f"{v:>14,.2f}" if v is not None else f"{'-':>14}"
    print(f"{rid:<13}{exp:>14,.2f}{f(a90)}{f(a_pay)}{f(a_pay2)}  {fc_s} {' '.join(marks)}")

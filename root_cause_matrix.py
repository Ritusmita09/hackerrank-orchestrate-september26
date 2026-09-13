"""Root-cause matrix for the 21 amount_safe_to_pay mismatches.

For each labeled sample, reverse-engineer which (mechanism, parameter) would
make the computed safe amount equal the expected value:

  - horizon H   : forecast_balance(state, days=H), H in {60,90,120,150,180,270,365}
  - expense pct : expense-pattern typical amounts at p25 / mean / p75 (horizon 90)
  - salary amt  : salary typical = most-recent / max historical (horizon 90)

Also reports: capped status, binding date (forecast minimum), and capped slack
for the four exact matches (how much extra conservatism they can absorb).
No production files are modified.
"""
import sys
import statistics
from pathlib import Path
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).parent / "code"))
import pandas as pd
from data_loader import DataLoader
from image_extractor import ImageExtractor
from financial_engine import FinancialEngine

loader = DataLoader()
data = loader.load_all()
lookups = loader.build_lookup_structures(data)

INCOME_CATS = {'salary', 'bonus', 'commission', 'freelance'}
SAMPLES = data['sample_requests'].to_dict('records')
EXACT_FOUR = ['request_01', 'request_09', 'request_12', 'request_16']

fe = FinancialEngine(data, lookups, ImageExtractor())


def safe(state, horizon):
    fc = fe.forecast_balance(state, days=horizon)
    return fe.calculate_safe_amount(state, state_req[state.user_id], fc)


# Precompute requested amounts by user+request (requests are unique per row)
state_req = {}


def pattern_amounts(state, pattern):
    by_id = {e['event_id']: e for e in state.historical_events + state.scheduled_events}
    return [float(e['amount']) for eid in pattern.event_ids
            if (e := by_id.get(eid)) is not None and e['amount'] is not None]


def with_expense_pct(state, q, horizon=90):
    """Safe amount with expense typical amounts at percentile q (None=mean)."""
    saved = [(p, p.typical_amount) for p in state.recurring_patterns]
    try:
        for p in state.recurring_patterns:
            if p.category not in INCOME_CATS:
                amts = pattern_amounts(state, p)
                if len(amts) >= 4:
                    try:
                        p.typical_amount = (statistics.mean(amts) if q == 'mean'
                                            else statistics.quantiles(amts, n=4)[{25: 0, 75: 2}[q]])
                    except Exception:
                        pass
        return safe(state, horizon)
    finally:
        for p, orig in saved:
            p.typical_amount = orig


def with_salary(state, mode, horizon=90):
    saved = [(p, p.typical_amount) for p in state.recurring_patterns]
    try:
        for p in state.recurring_patterns:
            if p.category == 'salary':
                amts = pattern_amounts(state, p)
                if amts:
                    p.typical_amount = max(amts) if mode == 'max' else amts[-1]
        return safe(state, horizon)
    finally:
        for p, orig in saved:
            p.typical_amount = orig


print(f"{'req':<12}{'diff':>14}  {'capped':>6} {'bindDate':>11} "
      f"{'H60':>10} {'H120':>10} {'H150':>10} {'H180':>10} {'H365':>10} "
      f"{'exp_p25':>10} {'exp_mean':>10} {'exp_p75':>10} {'sal_rec':>10}")

rows_out = []
for s in SAMPLES:
    rid = s['request_id']
    req_amt = float(s['requested_amount'])
    exp = float(s['amount_safe_to_pay'])
    state = fe.reconstruct_state(s['user_id'], pd.Timestamp(s['request_date']), s['request_id'])
    state_req[state.user_id] = req_amt  # set before safe() calls

    fc90 = fe.forecast_balance(state, days=90)
    base = fe.calculate_safe_amount(state, req_amt, fc90)
    capped = abs(base - req_amt) < 0.01
    bind_date = min(fc90, key=fc90.get)

    h = {hh: safe(state, hh) for hh in [60, 90, 120, 150, 180, 365]}
    ep = {q: with_expense_pct(state, q) for q in [25, 'mean', 75]}
    sr = with_salary(state, 'recent')

    diff = base - exp
    print(f"{rid:<12}{diff:>+14,.2f}  {str(capped):>6} {bind_date.date().isoformat():>11} "
          f"{h[60]:>10,.0f} {h[120]:>10,.0f} {h[150]:>10,.0f} {h[180]:>10,.0f} {h[365]:>10,.0f} "
          f"{ep[25]:>10,.0f} {ep['mean']:>10,.0f} {ep[75]:>10,.0f} {sr:>10,.0f}")

    # Which mechanism/parameter hits expected within 0.01?
    hits = []
    for hh, v in h.items():
        if abs(v - exp) < 0.01:
            hits.append(f'H={hh}')
    for q, v in ep.items():
        if abs(v - exp) < 0.01:
            hits.append(f'exp_{q}')
    if abs(sr - exp) < 0.01:
        hits.append('sal_recent')
    if abs(base - exp) < 0.01:
        hits.append('BASE')
    rows_out.append((rid, base, exp, hits))

print("\nExact-hit mechanisms per request:")
for rid, base, exp, hits in rows_out:
    if hits and hits != ['BASE']:
        print(f"  {rid}: {hits}")
    elif not hits:
        print(f"  {rid}: NONE of the tested mechanisms reaches expected ({exp:,.2f})")

# Capped slack for the four exact matches (conservatism headroom)
print("\nCapped slack for exact matches (min forecast balance - requested - min_balance):")
for s in SAMPLES:
    if s['request_id'] in EXACT_FOUR:
        req_amt = float(s['requested_amount'])
        state = fe.reconstruct_state(s['user_id'], pd.Timestamp(s['request_date']), s['request_id'])
        state_req[state.user_id] = req_amt
        for hh in [90, 120, 150]:
            fc = fe.forecast_balance(state, days=hh)
            slack = min(fc.values()) - req_amt - state.minimum_balance_to_keep
            print(f"  {s['request_id']} H={hh}: slack = {slack:,.2f}")
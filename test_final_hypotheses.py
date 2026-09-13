"""Final hypothesis battery for amount_safe_to_pay (NO production edits).

Request-level cash-flow decomposition + clustering, then in-memory tests of the
only mechanisms NOT covered by prior harnesses (test_rules, grid_forecast_rules,
test_final_rules, lifecycle audit):

  A. Boundary rules: exclude occurrences landing exactly on request_date
     (hypothesis: same-day expenses are already reflected in the balance).
  B. Dedup direction: count BOTH scheduled and pattern occurrences (dedup off),
     or drop scheduled events entirely.
  C. Minimum-balance buffer: minbal + fixed amount / percentage.
  D. Pending-debit placement: t0 vs settlement date (expected no-op; verified).

Everything replays cached production states through a configurable forecast
that mirrors production exactly under the production config (sanity-checked).
"""
import sys
import statistics
import contextlib
import io
from pathlib import Path
from datetime import timedelta
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent / 'code'))
import pandas as pd
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

samples = data['sample_requests'].to_dict('records')
EXPECTED = {s['request_id']: float(s['amount_safe_to_pay']) for s in samples}
REQUESTED = {s['request_id']: float(s['requested_amount']) for s in samples}
EXACT_FOUR = ['request_01', 'request_09', 'request_12', 'request_16']

STATES = {}
for s in samples:
    STATES[s['request_id']] = fe.reconstruct_state(
        s['user_id'], s['request_date'], s['request_id'])


# ============ 1. Decomposition / clustering ============
print("=" * 100)
print("REQUEST-LEVEL DECOMPOSITION (binding window, D_ref vs D_our, cluster)")
print("=" * 100)
rows = []
for s in samples:
    rid = s['request_id']
    exp, req = EXPECTED[rid], REQUESTED[rid]
    st = STATES[rid]
    if not (0 < exp < req):
        continue
    fc = fe.forecast_balance(st, days=90)
    bdate, bmin = min(fc.items(), key=lambda x: x[1])
    bal0 = st.current_available_balance
    pending = sum(p['amount'] for p in st.pending_debits)
    D_our = bal0 - pending - st.minimum_balance_to_keep - (bmin - st.minimum_balance_to_keep)
    # D_our = deduction our forecast applies before binding (incl. everything)
    D_our_full = bal0 - pending - bmin + st.minimum_balance_to_keep - st.minimum_balance_to_keep
    # simpler: D_our = bal0 - pending - bmin  ... define deduction = bal0 - pending - fmin_at_zero_payment
    # our amount (uncapped) = fmin - minbal -> D_our = bal0 - pending - fmin
    D_our = bal0 - pending - bmin
    D_ref = bal0 - pending - st.minimum_balance_to_keep - exp
    R = D_ref - D_our
    rows.append({
        'rid': rid, 'binding': str(bdate.date()), 'day': (bdate - st.as_of_date).days,
        'bal0': bal0, 'minbal': st.minimum_balance_to_keep, 'pending': pending,
        'D_our': D_our, 'D_ref': D_ref, 'R': R,
        'R_pct_D': (R / D_our * 100) if D_our else float('nan'),
        'R_pct_minbal': (R / st.minimum_balance_to_keep * 100),
        'R_pct_bal': (R / bal0 * 100),
        'sign': 'OVER' if R > 0 else 'UNDER',
    })
df = pd.DataFrame(rows)
pd.set_option('display.width', 220)
print(df.to_string(index=False, float_format=lambda x: f'{x:,.2f}'))
print()
print("Cluster A (reference deducts MORE, we are over-optimistic):",
      df[df.R > 0]['rid'].tolist())
print("Cluster B (reference deducts LESS, we are pessimistic):",
      df[df.R < 0]['rid'].tolist())
print()
for col in ['R_pct_D', 'R_pct_minbal', 'R_pct_bal']:
    over = df[df.R > 0][col]
    under = df[df.R < 0][col]
    print(f"{col}: OVER range [{over.min():+.2f}%, {over.max():+.2f}%] "
          f"UNDER range [{under.min():+.2f}%, {under.max():+.2f}%]  -> "
          f"{'POSSIBLE shared scale' if abs(over.max()-over.min()) < 2 and abs(under.max()-under.min()) < 2 else 'no consistent scale'}")


# ============ 2. Configurable forecast (mirrors production) ============
def build_forecast(state, cfg, days=95):
    rd = state.as_of_date
    end_date = rd + timedelta(days=days)
    minbal = state.minimum_balance_to_keep
    if cfg.get('minbal_add'):
        minbal = minbal + cfg['minbal_add']
    if cfg.get('minbal_mult'):
        minbal = minbal * cfg['minbal_mult']

    current_balance = state.current_available_balance
    if cfg.get('pending_at', 't0') == 't0':
        for p in state.pending_debits:
            current_balance -= p['amount']

    cash = defaultdict(float)
    dedup = cfg.get('dedup', 'prod')  # prod | off | no_sched

    if dedup != 'no_sched':
        for ev in state.scheduled_events:
            d = ev['settlement_date']
            if rd <= d <= end_date:
                cash[d] += ev['amount'] if ev['direction'] == 'credit' else -ev['amount']
    sched_keys = {(e['settlement_date'], e['category'], e['direction'])
                  for e in state.scheduled_events}

    exclude_rd_day = cfg.get('exclude_rd_day', False)

    for pattern in state.recurring_patterns:
        occurrences = fe.recurrence_detector.generate_future_occurrences(pattern, rd, days)
        for occ in occurrences:
            d = occ['settlement_date']
            if d > end_date:
                continue
            if exclude_rd_day and d == rd:
                continue
            if dedup == 'prod' and pattern.period_days == 0 \
                    and (d, occ['category'], occ['direction']) in sched_keys:
                continue
            sgn = 1 if occ['direction'] == 'credit' else -1
            cash[d] += sgn * occ['amount']

    if cfg.get('pending_at', 't0') == 'settlement':
        for p in state.pending_debits:
            d = p['settlement_date']
            if rd <= d <= end_date:
                cash[d] -= p['amount']

    forecast = {}
    bal = current_balance
    d = rd
    while d <= end_date:
        if d in cash:
            bal += cash[d]
        forecast[d] = bal
        d += timedelta(days=1)
    return forecast, minbal


def amounts_for(cfg):
    out = {}
    for s in samples:
        st = STATES[s['request_id']]
        fc, minbal = build_forecast(st, cfg)
        rd = st.as_of_date
        end = rd + timedelta(days=90)
        fmin = min(b for d, b in fc.items() if rd <= d <= end)
        amt = max(0.0, min(REQUESTED[s['request_id']], fmin - minbal))
        out[s['request_id']] = amt
    return out


PROD = {}
base = amounts_for(PROD)
base_n = sum(1 for r in base if abs(base[r] - EXPECTED[r]) < 0.01)
print()
print("=" * 100)
print(f"Sanity: custom forecast under production config: exact={base_n}/25")
for r in sorted(base):
    if abs(base[r] - EXPECTED[r]) >= 0.01:
        print(f"    {r}: {base[r]:,.2f} vs {EXPECTED[r]:,.2f} ({base[r]-EXPECTED[r]:+,.2f})")


def report(name, cfg):
    new = amounts_for(cfg)
    n = sum(1 for r in new if abs(new[r] - EXPECTED[r]) < 0.01)
    changed = [r for r in new if abs(new[r] - base[r]) > 0.005]
    imp, reg = [], []
    for r in new:
        eb, en = abs(base[r] - EXPECTED[r]), abs(new[r] - EXPECTED[r])
        if en < eb - 0.005:
            imp.append((r, base[r], new[r], EXPECTED[r]))
        elif en > eb + 0.005:
            reg.append((r, base[r], new[r], EXPECTED[r]))
    four_ok = all(abs(new[r] - EXPECTED[r]) < 0.01 for r in EXACT_FOUR)
    print(f"\n=== {name}: exact {base_n}/25 -> {n}/25 | exact-four intact: {four_ok} ===")
    print(f"  changed: {len(changed)}")
    for r, b, v, e in imp:
        print(f"    IMPROVED {r}: {b:,.2f} -> {v:,.2f} (exp {e:,.2f})")
    for r, b, v, e in reg:
        print(f"    REGRESSED {r}: {b:,.2f} -> {v:,.2f} (exp {e:,.2f})")
    return n


print()
print("=" * 100)
print("HYPOTHESIS BATTERY (in-memory)")
print("=" * 100)
report("A1: exclude occurrences ON request_date", dict(exclude_rd_day=True))
report("B1: dedup OFF (count scheduled AND pattern)", dict(dedup='off'))
report("B2: no scheduled events in forecast", dict(dedup='no_sched'))
report("D1: pending debits at settlement date", dict(pending_at='settlement'))
for buf in (10, 25, 50, 100, 250, 500):
    report(f"C1: minbal + {buf}", dict(minbal_add=buf))
for m in (1.01, 1.02, 1.05):
    report(f"C2: minbal x {m}", dict(minbal_mult=m))

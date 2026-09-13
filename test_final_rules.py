"""Final in-memory rule testing for amount_safe_to_pay (no production edits).

Families NOT covered by earlier harnesses (test_rules.py: percentile/mean/horizon
rules; grid_forecast_rules.py: stat x cadence x salary x pending-timing grid):
  A. Integer rounding of forecast occurrence amounts (round/ceil/floor of the
     median typical, and of the mean) — motivated by the forensic finding that
     the reference's total deductions are INTEGERS in 16/17 interior mismatches.
  B. Flexibility-based deductions: dropping stoppable / flexible patterns from
     the forecast, or forecasting reducible patterns at minimum_allowed_amount.

For each candidate rule, all 25 labeled samples are recomputed through the real
process_request pipeline (with a patched reconstruct_state that replays cached
states and applies the rule in-memory), and we report:
  1. amount_safe_to_pay exact matches before/after,
  2. full 6-field matches before/after,
  3. every request changed,
  4. regressions (previously-matching requests broken, or error increased).
"""
import sys
import io
import copy
import statistics
import contextlib
from pathlib import Path

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

INCOME_CATS = {'salary', 'bonus', 'commission', 'freelance'}
EXACT_FOUR = ['request_01', 'request_09', 'request_12', 'request_16']
SAMPLES = data['sample_requests'].to_dict('records')
EXPECTED = {s['request_id']: float(s['amount_safe_to_pay']) for s in SAMPLES}
REQUESTED = {s['request_id']: float(s['requested_amount']) for s in SAMPLES}

FIELDS = ['amount_safe_to_pay', 'affordability_status', 'recommended_payment_method',
          'payment_plan', 'earliest_date_for_full_payment', 'spending_changes_needed']


def compare_field(actual, expected, field_name):
    actual_str = str(actual).strip() if pd.notna(actual) else ''
    expected_str = str(expected).strip() if pd.notna(expected) else ''
    if actual_str == expected_str:
        return True
    if field_name == 'earliest_date_for_full_payment':
        try:
            if actual_str and expected_str:
                if pd.to_datetime(actual_str).date() == pd.to_datetime(expected_str).date():
                    return True
        except Exception:
            pass
    if field_name == 'amount_safe_to_pay':
        try:
            if abs(float(actual_str) - float(expected_str)) < 0.01:
                return True
        except Exception:
            pass
    return False


def build_engines():
    fe = FinancialEngine(data, lookups, ImageExtractor())
    de = DecisionEngine(fe)
    return fe, de


# ---- cache reconstructed states once (production behavior) ----
fe0, de0 = build_engines()
STATE_CACHE = {}
for s in SAMPLES:
    STATE_CACHE[s['request_id']] = fe0.reconstruct_state(
        s['user_id'], pd.Timestamp(s['request_date']), s['request_id'])


def run_all(rule=None):
    """Run all 25 samples through the real pipeline with an optional pattern rule."""
    fe, de = build_engines()
    orig = fe.reconstruct_state

    def patched(user_id, request_date, request_id=None):
        # replay the cached production state (identical reconstruction);
        # deep-copy so in-memory rules never contaminate the cache
        if request_id in STATE_CACHE:
            state = copy.deepcopy(STATE_CACHE[request_id])
        else:
            state = orig(user_id, request_date, request_id)
        if rule is not None:
            rule(state)
        return state

    fe.reconstruct_state = patched
    out = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for s in SAMPLES:
            o = process_request(s, data, lookups, fe, de)
            full = all(compare_field(o[f], s[f], f) for f in FIELDS)
            out[s['request_id']] = {
                'amount': float(o['amount_safe_to_pay']),
                'full': full,
            }
    return out


def hist_amounts(state, pattern):
    direction = 'credit' if pattern.category in INCOME_CATS else 'debit'
    evs = sorted([e for e in state.historical_events
                  if e['category'] == pattern.category and e['direction'] == direction],
                 key=lambda e: e['settlement_date'])
    return [float(e['amount']) for e in evs if e['amount'] is not None]


# ---------------- Candidate rules ----------------

def _expense_patterns(state):
    return [p for p in state.recurring_patterns if p.category not in INCOME_CATS]


def rule_expense_round(state):
    for p in _expense_patterns(state):
        p.typical_amount = float(round(p.typical_amount))


def rule_expense_ceil(state):
    for p in _expense_patterns(state):
        v = p.typical_amount
        p.typical_amount = float(int(v) + (1 if v % 1 else 0))


def rule_expense_floor(state):
    for p in _expense_patterns(state):
        p.typical_amount = float(int(p.typical_amount))


def rule_expense_round_mean(state):
    for p in _expense_patterns(state):
        amts = hist_amounts(state, p)
        if amts:
            p.typical_amount = float(round(sum(amts) / len(amts)))


def rule_expense_ceil_mean(state):
    for p in _expense_patterns(state):
        amts = hist_amounts(state, p)
        if amts:
            m = sum(amts) / len(amts)
            p.typical_amount = float(int(m) + (1 if m % 1 else 0))


def rule_var_round(state):
    for p in _expense_patterns(state):
        if p.period_days > 0:
            p.typical_amount = float(round(p.typical_amount))


def rule_var_ceil(state):
    for p in _expense_patterns(state):
        if p.period_days > 0:
            v = p.typical_amount
            p.typical_amount = float(int(v) + (1 if v % 1 else 0))


def rule_mon_round(state):
    for p in _expense_patterns(state):
        if p.period_days == 0:
            p.typical_amount = float(round(p.typical_amount))


def rule_var_round_mean(state):
    for p in _expense_patterns(state):
        if p.period_days > 0:
            amts = hist_amounts(state, p)
            if amts:
                p.typical_amount = float(round(sum(amts) / len(amts)))


def rule_drop_stoppable(state):
    state.recurring_patterns = [p for p in state.recurring_patterns
                                if p.flexibility != 'stoppable']


def rule_drop_flexible(state):
    state.recurring_patterns = [p for p in state.recurring_patterns
                                if p.flexibility not in ('stoppable', 'reducible')]


def rule_reducible_at_min(state):
    for p in _expense_patterns(state):
        if p.flexibility == 'reducible' and p.minimum_allowed_amount is not None:
            p.typical_amount = float(p.minimum_allowed_amount)


def rule_flexible_at_min(state):
    for p in _expense_patterns(state):
        if p.flexibility in ('stoppable', 'reducible') and p.minimum_allowed_amount is not None:
            p.typical_amount = float(p.minimum_allowed_amount)


RULES = [
    ("expense_round (typ -> round(median))", rule_expense_round),
    ("expense_ceil (typ -> ceil(median))", rule_expense_ceil),
    ("expense_floor (typ -> floor(median))", rule_expense_floor),
    ("expense_round_mean", rule_expense_round_mean),
    ("expense_ceil_mean", rule_expense_ceil_mean),
    ("var_round (interval patterns only)", rule_var_round),
    ("var_ceil (interval patterns only)", rule_var_ceil),
    ("mon_round (monthly patterns only)", rule_mon_round),
    ("var_round_mean (interval patterns only)", rule_var_round_mean),
    ("drop_stoppable", rule_drop_stoppable),
    ("drop_flexible", rule_drop_flexible),
    ("reducible_at_min", rule_reducible_at_min),
    ("flexible_at_min", rule_flexible_at_min),
]


def run_with_horizon(horizon):
    """Run all 25 samples with a different forecast horizon."""
    fe, de = build_engines()
    orig = fe.reconstruct_state

    def patched(user_id, request_date, request_id=None):
        if request_id in STATE_CACHE:
            state = copy.deepcopy(STATE_CACHE[request_id])
        else:
            state = orig(user_id, request_date, request_id)
        return state

    fe.reconstruct_state = patched
    orig_fc = fe.forecast_balance
    fe.forecast_balance = lambda state, days=90: orig_fc(state, horizon)
    out = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for s in SAMPLES:
            o = process_request(s, data, lookups, fe, de)
            full = all(compare_field(o[f], s[f], f) for f in FIELDS)
            out[s['request_id']] = {'amount': float(o['amount_safe_to_pay']), 'full': full}
    return out


def report(name, base, new):
    b_amt = sum(1 for r in base if abs(base[r]['amount'] - EXPECTED[r]) < 0.01)
    n_amt = sum(1 for r in new if abs(new[r]['amount'] - EXPECTED[r]) < 0.01)
    b_full = sum(1 for r in base if base[r]['full'])
    n_full = sum(1 for r in new if new[r]['full'])
    changed = [r for r in base if abs(base[r]['amount'] - new[r]['amount']) > 0.005]
    regressions = []
    improvements = []
    for r in base:
        eb = abs(base[r]['amount'] - EXPECTED[r])
        en = abs(new[r]['amount'] - EXPECTED[r])
        if en > eb + 0.005:
            regressions.append((r, base[r]['amount'], new[r]['amount'], EXPECTED[r]))
        elif en < eb - 0.005:
            improvements.append((r, base[r]['amount'], new[r]['amount'], EXPECTED[r]))
    exact_ok = all(abs(new[r]['amount'] - EXPECTED[r]) < 0.01 and new[r]['full']
                   for r in EXACT_FOUR)
    print(f"\n=== RULE: {name} ===")
    print(f"  amount matches: {b_amt}/25 -> {n_amt}/25 | full matches: {b_full}/25 -> {n_full}/25")
    print(f"  exact-four (01/09/12/16) still full-match: {exact_ok}")
    print(f"  changed ({len(changed)}): {changed}")
    if improvements:
        print(f"  improved ({len(improvements)}):")
        for r, b, n, e in improvements:
            print(f"    {r}: {b:,.2f} -> {n:,.2f} (exp {e:,.2f})")
    if regressions:
        print(f"  REGRESSIONS ({len(regressions)}):")
        for r, b, n, e in regressions:
            print(f"    {r}: {b:,.2f} -> {n:,.2f} (exp {e:,.2f})")


if __name__ == '__main__':
    print("Computing baseline (cached states, no rule)...")
    base = run_all(None)
    b_amt = sum(1 for r in base if abs(base[r]['amount'] - EXPECTED[r]) < 0.01)
    b_full = sum(1 for r in base if base[r]['full'])
    print(f"Baseline: amount matches {b_amt}/25, full matches {b_full}/25")
    for r in sorted(base):
        d = base[r]['amount'] - EXPECTED[r]
        flag = 'EXACT' if abs(d) < 0.01 else ('OVER ' if d > 0 else 'UNDER')
        cap = ' [capped]' if abs(base[r]['amount'] - REQUESTED[r]) < 0.01 else ''
        print(f"  {r}: {base[r]['amount']:>15,.2f} vs {EXPECTED[r]:>15,.2f}  {d:>+14,.2f} {flag}{cap}")

    # ---- D_ref integrality evidence ----
    print("\n=== Reference deduction integrality (interior mismatches) ===")
    for s in SAMPLES:
        rid = s['request_id']
        exp, req = EXPECTED[rid], REQUESTED[rid]
        if not (0 < exp < req):
            continue
        st = STATE_CACHE[rid]
        bal0 = st.current_available_balance
        pending = sum(p['amount'] for p in st.pending_debits)
        d_ref = bal0 - pending - st.minimum_balance_to_keep - exp
        print(f"  {rid}: D_ref = {d_ref:,.2f}  {'INTEGER' if abs(d_ref - round(d_ref)) < 0.005 else 'non-integer'}")

    for name, fn in RULES:
        print(f"\nRunning rule: {name} ...")
        new = run_all(fn)
        report(name, base, new)

    for h in (120, 150):
        print(f"\nRunning rule: horizon_{h} ...")
        new = run_with_horizon(h)
        report(f"horizon_{h} (forecast {h} days)", base, new)

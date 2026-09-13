"""In-memory rule testing for amount_safe_to_pay mismatches.

No production files are modified. Each candidate rule is applied by wrapping
fe.reconstruct_state and mutating the resulting state's recurring patterns
(or by wrapping fe.forecast_balance for horizon rules), then all 25 labeled
samples are recomputed and compared to baseline.

Key structural fact discovered: all four exact matches (01, 09, 12, 16) have
computed safe amount == requested amount (capped). Making the model LESS
conservative cannot push them above the cap, so they are structurally safe
from less-conservative rules; MORE conservative rules risk breaking them.
"""
import sys
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "code"))
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
LARGE_ERROR = ['request_05', 'request_07', 'request_10', 'request_15', 'request_20', 'request_25']
EXACT_FOUR = ['request_01', 'request_09', 'request_12', 'request_16']

SAMPLES = data['sample_requests'].to_dict('records')
EXPECTED = {s['request_id']: float(s['amount_safe_to_pay']) for s in SAMPLES}
REQUESTED = {s['request_id']: float(s['requested_amount']) for s in SAMPLES}


def build_engines():
    fe = FinancialEngine(data, lookups, ImageExtractor())
    de = DecisionEngine(fe)
    return fe, de


def compute_all(fe, de):
    out = {}
    for s in SAMPLES:
        o = process_request(s, data, lookups, fe, de)
        out[s['request_id']] = float(o['amount_safe_to_pay'])
    return out


def run_with_pattern_rule(rule):
    """Run all 25 samples with a rule that mutates state.recurring_patterns."""
    fe, de = build_engines()
    orig = fe.reconstruct_state

    def patched(user_id, request_date, request_id=None):
        state = orig(user_id, request_date, request_id)
        rule(state)
        return state

    fe.reconstruct_state = patched
    return compute_all(fe, de)


def run_with_horizon(horizon):
    """Run all 25 samples with a different forecast horizon."""
    fe, de = build_engines()
    orig = fe.forecast_balance
    fe.forecast_balance = lambda state, days=90: orig(state, horizon)
    return compute_all(fe, de)


def run_baseline():
    fe, de = build_engines()
    return compute_all(fe, de)


def pattern_amounts(state, pattern):
    """Historical amounts of the events backing a pattern."""
    by_id = {e['event_id']: e for e in state.historical_events + state.scheduled_events}
    amts = []
    for eid in pattern.event_ids:
        e = by_id.get(eid)
        if e is not None and e['amount'] is not None:
            amts.append(float(e['amount']))
    return amts


def pct(amts, q):
    """Safely get a percentile of amounts; None if not enough data."""
    if len(amts) < 4:
        return None
    try:
        if q == 25:
            return statistics.quantiles(amts, n=4)[0]
        if q == 75:
            return statistics.quantiles(amts, n=4)[2]
    except Exception:
        return None
    return None


# ---------------- Candidate rules ----------------

def rule_income_median(state):
    """Cat 2: all income patterns use median of historical amounts (less
    conservative than 10th-pct/min used for non-salary income)."""
    for p in state.recurring_patterns:
        if p.category in INCOME_CATS:
            amts = pattern_amounts(state, p)
            if len(amts) >= 2:
                p.typical_amount = statistics.median(amts)


def rule_expense_p25(state):
    """Cat 1: expense typical amount = 25th percentile (less conservative)."""
    for p in state.recurring_patterns:
        if p.category not in INCOME_CATS:
            v = pct(pattern_amounts(state, p), 25)
            if v is not None:
                p.typical_amount = v


def rule_expense_p75(state):
    """Cat 1: expense typical amount = 75th percentile (more conservative)."""
    for p in state.recurring_patterns:
        if p.category not in INCOME_CATS:
            v = pct(pattern_amounts(state, p), 75)
            if v is not None:
                p.typical_amount = v


def rule_expense_mean(state):
    """Cat 1: expense typical amount = mean (captures infrequent large items)."""
    for p in state.recurring_patterns:
        if p.category not in INCOME_CATS:
            amts = pattern_amounts(state, p)
            if len(amts) >= 2:
                p.typical_amount = statistics.mean(amts)


def rule_salary_max(state):
    """Cat 2: salary typical amount = max historical (aggressive income)."""
    for p in state.recurring_patterns:
        if p.category == 'salary':
            amts = pattern_amounts(state, p)
            if amts:
                p.typical_amount = max(amts)


def rule_expense_p25_variable_only(state):
    """Cat 1 (targeted): 25th percentile only for weekly/interval expense
    patterns (groceries/transport/dining), monthly fixed ones untouched."""
    for p in state.recurring_patterns:
        if p.category not in INCOME_CATS and p.period_days > 0:
            v = pct(pattern_amounts(state, p), 25)
            if v is not None:
                p.typical_amount = v


# ---------------- Report ----------------

def classify(base, new, rid):
    eb, en = abs(base[rid] - EXPECTED[rid]), abs(new[rid] - EXPECTED[rid])
    if en < eb - 0.005:
        return 'improve'
    if en > eb + 0.005:
        return 'worsen'
    return 'same'


def report(name, base, new):
    results = {rid: classify(base, new, rid) for rid in EXPECTED}
    improved = [r for r, v in results.items() if v == 'improve']
    worsened = [r for r, v in results.items() if v == 'worsen']
    exact_ok = all(abs(new[r] - EXPECTED[r]) < 0.01 for r in EXACT_FOUR)
    print(f"\n=== RULE: {name} ===")
    print(f"  improved ({len(improved)}): {improved}")
    print(f"  worsened ({len(worsened)}): {worsened}")
    print(f"  01/09/12/16 remain exact: {exact_ok}")
    print("  Large-error cases:")
    for rid in LARGE_ERROR:
        print(f"    {rid}: base {base[rid]:,.2f} -> new {new[rid]:,.2f} "
              f"(exp {EXPECTED[rid]:,.2f}) [{results[rid]}]")


if __name__ == '__main__':
    print("Computing baseline...")
    base = run_baseline()
    print("\nBaseline (actual vs expected):")
    for s in SAMPLES:
        rid = s['request_id']
        d = base[rid] - EXPECTED[rid]
        flag = 'EXACT' if abs(d) < 0.01 else ('OVER' if d > 0 else 'UNDER')
        cap = ' [capped=req]' if abs(base[rid] - REQUESTED[rid]) < 0.01 else ''
        print(f"  {rid}: {base[rid]:>15,.2f} vs {EXPECTED[rid]:>15,.2f}  {d:>+14,.2f} {flag}{cap}")

    rules = [
        ("income_median (all income -> median)", lambda: run_with_pattern_rule(rule_income_median)),
        ("expense_p25 (expenses -> 25th pct)", lambda: run_with_pattern_rule(rule_expense_p25)),
        ("expense_p75 (expenses -> 75th pct)", lambda: run_with_pattern_rule(rule_expense_p75)),
        ("expense_mean (expenses -> mean)", lambda: run_with_pattern_rule(rule_expense_mean)),
        ("salary_max (salary -> max historical)", lambda: run_with_pattern_rule(rule_salary_max)),
        ("expense_p25_variable_only (interval expenses -> 25th pct)",
         lambda: run_with_pattern_rule(rule_expense_p25_variable_only)),
        ("horizon_60 (forecast 60 days)", lambda: run_with_horizon(60)),
        ("horizon_180 (forecast 180 days)", lambda: run_with_horizon(180)),
    ]
    for name, fn in rules:
        print(f"\nRunning rule: {name} ...")
        new = fn()
        report(name, base, new)
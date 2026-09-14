"""Typed domain models for the deterministic financial engine.

Design rules enforced here:

* **Money is integer minor units + ISO-4217 currency code.** ``Decimal`` is
  accepted at the edges (``Money.from_major``) but arithmetic inside the
  engine operates on ``int`` minor units exclusively. Floats are rejected on
  construction. This eliminates an entire class of rounding defects.
* **Everything is a frozen dataclass or an Enum.** No loosely-typed
  dictionaries travel through the core.
* **No I/O, no clock.** Models are passive value objects; time and data are
  always supplied by the caller.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Errors — the engine never fails silently and never fabricates values.
# ---------------------------------------------------------------------------


class EngineError(Exception):
    """Base class for all deterministic-engine errors."""


class InvalidMoneyError(EngineError):
    """A monetary value was malformed, imprecise, or a float."""


class CurrencyMismatchError(EngineError):
    """Arithmetic or comparison between different currencies."""


class CurrencyNotNormalizedError(EngineError):
    """An event's currency is not the user's home currency.

    Currency normalization (via dated FX rates) is an ingestion-pipeline
    responsibility that happens BEFORE the engine. The engine performs
    single-currency arithmetic only and refuses mixed currencies loudly.
    """


class MissingAmountError(EngineError):
    """A flow-relevant event has no amount and no way to derive one.

    The engine never invents amounts (no silent zero-fill, no guessing). The
    caller must resolve the amount (e.g. via document extraction + user
    confirmation) before building state.
    """


class DuplicateEventIdError(EngineError):
    """Two events share an event id — the ledger is ambiguous."""


class InvalidRequestError(EngineError):
    """An affordability request is internally inconsistent."""


# ---------------------------------------------------------------------------
# Money
# ---------------------------------------------------------------------------

#: Currencies with no minor unit (ISO 4217). All others are assumed to have
#: 2 decimal places, which covers every MVP-supported currency.
ZERO_DECIMAL_CURRENCIES = frozenset({
    "BIF", "CLP", "DJF", "GNF", "ISK", "JPY", "KMF", "KRW",
    "PYG", "RWF", "UGX", "VND", "VUV", "XAF", "XOF", "XPF",
})


def currency_exponent(currency: str) -> int:
    """Return the number of minor-unit digits for ``currency`` (0 or 2)."""
    return 0 if currency in ZERO_DECIMAL_CURRENCIES else 2


@dataclass(frozen=True)
class Money:
    """An amount of money in integer minor units, with an ISO currency code.

    Examples (2-decimal currency)::

        Money.from_major("123.45", "USD") == Money.from_minor(12345, "USD")
        Money.from_major("1000", "JPY")   == Money.from_minor(1000, "JPY")

    Floats are rejected outright — ``Money.from_major(0.1 + 0.2, "USD")``
    raises :class:`InvalidMoneyError` rather than silently corrupting a
    financial figure.
    """

    amount_minor: int
    currency: str

    def __post_init__(self) -> None:
        if not isinstance(self.amount_minor, int) or isinstance(self.amount_minor, bool):
            raise InvalidMoneyError(
                f"amount_minor must be int, got {type(self.amount_minor).__name__}"
            )
        if not isinstance(self.currency, str) or len(self.currency) != 3 \
                or not self.currency.isalpha() or not self.currency.isupper():
            raise InvalidMoneyError(
                f"currency must be a 3-letter uppercase ISO code, got {self.currency!r}"
            )

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_major(cls, value: Union[Decimal, str, int], currency: str) -> "Money":
        """Build from a major-unit amount given as Decimal, str, or int.

        Raises:
            InvalidMoneyError: if ``value`` is a float, unparseable, or has
                more precision than the currency supports (e.g. "1.005" USD).
        """
        if isinstance(value, float):
            raise InvalidMoneyError(
                "floats are never accepted for money; use Money.from_major(str(x)) "
                "or Decimal(x) explicitly at the boundary"
            )
        if isinstance(value, int):
            decimal_value = Decimal(value)
        elif isinstance(value, (str, Decimal)):
            try:
                decimal_value = Decimal(value)
            except InvalidOperation as exc:
                raise InvalidMoneyError(f"unparseable amount {value!r}") from exc
        else:
            raise InvalidMoneyError(f"unsupported amount type {type(value).__name__}")
        exponent = currency_exponent(currency)
        scaled = decimal_value.scaleb(exponent)
        if scaled != scaled.to_integral_value():
            raise InvalidMoneyError(
                f"amount {value!r} has more precision than {currency} "
                f"({exponent} decimal places) supports"
            )
        return cls(int(scaled), currency)

    @classmethod
    def from_minor(cls, amount_minor: int, currency: str) -> "Money":
        """Build from an integer amount of minor units."""
        return cls(amount_minor, currency)

    @classmethod
    def zero(cls, currency: str) -> "Money":
        """Return the zero amount of ``currency``."""
        return cls(0, currency)

    # -- conversion ---------------------------------------------------------

    def to_major(self) -> Decimal:
        """Return the amount in major units as an exact Decimal."""
        return Decimal(self.amount_minor).scaleb(-currency_exponent(self.currency))

    def __str__(self) -> str:
        return f"{self.currency} {self.to_major()}"

    __repr__ = __str__

    # -- arithmetic ---------------------------------------------------------

    def _require_same_currency(self, other: "Money") -> None:
        if not isinstance(other, Money):
            raise InvalidMoneyError(f"not a Money value: {other!r}")
        if other.currency != self.currency:
            raise CurrencyMismatchError(
                f"cannot combine {self.currency} with {other.currency}"
            )

    def __add__(self, other: "Money") -> "Money":
        self._require_same_currency(other)
        return Money(self.amount_minor + other.amount_minor, self.currency)

    def __sub__(self, other: "Money") -> "Money":
        self._require_same_currency(other)
        return Money(self.amount_minor - other.amount_minor, self.currency)

    def __neg__(self) -> "Money":
        return Money(-self.amount_minor, self.currency)

    def __abs__(self) -> "Money":
        return Money(abs(self.amount_minor), self.currency)

    # -- comparison (currency-checked) ---------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        self._require_same_currency(other)
        return self.amount_minor == other.amount_minor

    def __lt__(self, other: "Money") -> bool:
        self._require_same_currency(other)
        return self.amount_minor < other.amount_minor

    def __le__(self, other: "Money") -> bool:
        self._require_same_currency(other)
        return self.amount_minor <= other.amount_minor

    def __gt__(self, other: "Money") -> bool:
        self._require_same_currency(other)
        return self.amount_minor > other.amount_minor

    def __ge__(self, other: "Money") -> bool:
        self._require_same_currency(other)
        return self.amount_minor >= other.amount_minor

    def __hash__(self) -> int:
        return hash((self.amount_minor, self.currency))

    # -- predicates -----------------------------------------------------------

    @property
    def is_zero(self) -> bool:
        return self.amount_minor == 0

    @property
    def is_positive(self) -> bool:
        return self.amount_minor > 0

    @property
    def is_negative(self) -> bool:
        return self.amount_minor < 0


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Direction(enum.Enum):
    """Direction of a cash flow: money in (credit) or money out (debit)."""

    CREDIT = "credit"
    DEBIT = "debit"


class EventStatus(enum.Enum):
    """Lifecycle status of a financial event.

    Forecast semantics (see forecast.py):

    * ``SETTLED``   — already happened; history.
    * ``PENDING``   — announced but not settled. Debits are *reserved*
      (conservatively deducted) at the start of the forecast; credits are
      *excluded* (never counted as available money).
    * ``SCHEDULED`` — confirmed future flow; counted in the forecast.
    * ``FORECASTED``— generated by a recurring pattern; counted in the
      forecast, deduplicated against scheduled events by stream.
    * ``CANCELLED`` / ``FAILED`` — excluded from all flows.
    * ``UNREALIZED``— an investment valuation, not cash; excluded.
    """

    SETTLED = "settled"
    PENDING = "pending"
    SCHEDULED = "scheduled"
    FORECASTED = "forecasted"
    CANCELLED = "cancelled"
    FAILED = "failed"
    UNREALIZED = "unrealized"


class Flexibility(enum.Enum):
    """Whether a recurring expense can be changed by a spending change."""

    FIXED = "fixed"          # cannot be stopped or reduced
    REDUCIBLE = "reducible"  # can be reduced, down to min_allowed_amount
    STOPPABLE = "stoppable"  # can be stopped entirely


class PaymentMethod(enum.Enum):
    """Payment strategy a plan can use."""

    FULL_PAYMENT = "full_payment"
    PARTIAL_PAYMENT = "partial_payment"
    INSTALLMENTS = "installments"
    WAIT = "wait"
    NOT_RECOMMENDED = "not_recommended"


class AffordabilityStatus(enum.Enum):
    """Overall verdict for an affordability request."""

    AFFORDABLE_NOW = "affordable_now"
    AFFORDABLE_WITH_PLAN = "affordable_with_plan"
    AFFORDABLE_LATER = "affordable_later"
    NOT_AFFORDABLE = "not_affordable"


class SourceType(enum.Enum):
    """Where a fact came from (provenance)."""

    MANUAL = "manual"
    CSV = "csv"
    DOCUMENT = "document"
    MESSAGE = "message"
    PATTERN = "pattern"
    IMPORT = "import"


class Estimator(enum.Enum):
    """How a recurring pattern's typical amount was estimated.

    Income is estimated conservatively (low percentile / minimum) so that
    forecasts never overstate available money; expenses use the median.
    """

    MIN = "min"
    P10 = "p10"
    MEDIAN = "median"


class SpendingChangeAction(enum.Enum):
    """Action applied to a recurring expense stream."""

    STOP = "stop"                # stop the stream entirely
    REDUCE_TO = "reduce_to"      # reduce each occurrence to new_amount


class ViolationCode(enum.Enum):
    """Machine-readable reason a plan failed verification."""

    MIN_BALANCE_BREACH = "min_balance_breach"
    DEADLINE_MISSED = "deadline_missed"
    METHOD_NOT_PERMITTED = "method_not_permitted"
    PARTIAL_NOT_PERMITTED = "partial_not_permitted"
    INSTALLMENT_LIMIT_EXCEEDED = "installment_limit_exceeded"
    INSTALLMENT_MISMATCH = "installment_mismatch"
    PAYMENT_STRUCTURE_INVALID = "payment_structure_invalid"
    OBLIGATION_NOT_SATISFIED = "obligation_not_satisfied"
    DATE_BEFORE_AS_OF = "date_before_as_of"
    DATE_BEYOND_HORIZON = "date_beyond_horizon"
    NEGATIVE_OR_ZERO_PAYMENT = "negative_or_zero_payment"
    CURRENCY_MISMATCH = "currency_mismatch"
    SPENDING_CHANGE_UNAUTHORIZED = "spending_change_unauthorized"
    SPENDING_CHANGE_INVALID = "spending_change_invalid"


class EvidenceType(enum.Enum):
    """Kind of structured evidence attached to a decision."""

    REQUEST = "request"
    BALANCE = "balance"
    FORECAST_POINT = "forecast_point"
    MIN_BALANCE_FLOOR = "min_balance_floor"
    PAYMENT = "payment"
    SPENDING_CHANGE = "spending_change"
    PATTERN = "pattern"
    CONFLICT_RESOLUTION = "conflict_resolution"
    SCHEDULED_EVENT = "scheduled_event"
    SAFE_AMOUNT = "safe_amount"
    EARLIEST_DATE = "earliest_date"


class Impact(enum.Enum):
    """How decisive a piece of evidence was for the final decision."""

    DECISIVE = "decisive"
    SUPPORTING = "supporting"
    CONTEXTUAL = "contextual"


# ---------------------------------------------------------------------------
# Provenance and events
# ---------------------------------------------------------------------------


def normalize_description(description: str) -> str:
    """Normalize an event description for stream grouping.

    Case- and whitespace-insensitive so "ACME Payroll" and "acme  payroll "
    group to the same stream, while different merchants stay separate.
    """
    return " ".join(description.strip().lower().split())


@dataclass(frozen=True)
class Provenance:
    """Chain of custody for one fact."""

    source_type: SourceType
    source_id: str
    confidence: float = 1.0
    confirmed_by_user: bool = True


@dataclass(frozen=True)
class FinancialEvent:
    """A single normalized financial event in the user's ledger.

    ``amount`` is Optional only to represent not-yet-resolved drafts at the
    ingestion boundary; :func:`app.core.state.build_state` refuses to build
    a state containing a flow-relevant event without an amount.
    """

    event_id: str
    date: date
    direction: Direction
    amount: Optional[Money]
    category: str
    description: str = ""
    status: EventStatus = EventStatus.SETTLED
    flexibility: Flexibility = Flexibility.FIXED
    min_allowed_amount: Optional[Money] = None
    provenance: Provenance = field(
        default_factory=lambda: Provenance(SourceType.MANUAL, "unknown")
    )
    linked_event_id: Optional[str] = None
    amends_event_id: Optional[str] = None

    def stream_key(self) -> str:
        """Identity of the recurring stream this event belongs to.

        Groups by (normalized description, category, direction). This is the
        fix for the old system's failure mode where two different salary
        streams merged into one pattern because grouping ignored description.
        """
        return f"{normalize_description(self.description)}|{self.category}|{self.direction.value}"


# ---------------------------------------------------------------------------
# Recurrence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedPeriod:
    """A recurrence every ``days`` days (e.g. weekly = 7, biweekly = 14)."""

    days: int

    def __post_init__(self) -> None:
        if self.days < 1:
            raise EngineError(f"FixedPeriod.days must be >= 1, got {self.days}")

    def occurrences(self, start: date, end: date, last_seen: date) -> Tuple[date, ...]:
        """All occurrence dates in ``[start, end]`` strictly after ``last_seen``."""
        if end < start or end <= last_seen:
            return ()
        # First occurrence: smallest last_seen + k*days (k >= 1) that is >= start.
        # Starting from last_seen + days guarantees the series never
        # regenerates the historical event itself.
        current = last_seen + timedelta(days=self.days)
        while current < start:
            current = current + timedelta(days=self.days)
        result: list[date] = []
        while current <= end:
            result.append(current)
            current = current + timedelta(days=self.days)
        return tuple(result)


@dataclass(frozen=True)
class MonthlyPeriod:
    """A recurrence anchored to specific day(s) of the month.

    One anchor day is a monthly pattern; two anchor days (e.g. the 15th and
    the last day) is semi-monthly. Days beyond a month's length are clamped
    to the month's last day: an anchor of 31 lands on Feb 28 (29 in leap
    years), Apr 30, etc. Clamped dates are monotone non-decreasing, so a
    generated series can never step backward.
    """

    days_of_month: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.days_of_month:
            raise EngineError("MonthlyPeriod requires at least one anchor day")
        for day in self.days_of_month:
            if not 1 <= day <= 31:
                raise EngineError(f"anchor day must be in 1..31, got {day}")
        if len(set(self.days_of_month)) != len(self.days_of_month):
            raise EngineError("duplicate anchor days")

    @staticmethod
    def _clamp(year: int, month: int, day: int) -> date:
        import calendar
        last_day = calendar.monthrange(year, month)[1]
        return date(year, month, min(day, last_day))

    def occurrences(self, start: date, end: date, last_seen: date) -> Tuple[date, ...]:
        """All occurrence dates in ``[start, end]`` strictly after ``last_seen``."""
        if end < start or end <= last_seen:
            return ()
        result: list[date] = []
        year, month = start.year, start.month
        # Scan forward month by month; hard bound of 1200 months (~100 years).
        for _ in range(1200):
            for day in self.days_of_month:
                candidate = self._clamp(year, month, day)
                if start <= candidate <= end and candidate > last_seen:
                    result.append(candidate)
            if date(year, month, 1) > end:
                break
            month += 1
            if month > 12:
                month, year = 1, year + 1
        return tuple(sorted(result))


@dataclass(frozen=True)
class RecurringPattern:
    """A detected (or user-confirmed) recurring stream.

    A detected pattern is a *proposal* (``user_confirmed=False``) until the
    user confirms it; the confidence score travels with every occurrence so
    downstream consumers can decide how much to trust it.
    """

    pattern_id: str
    stream_key: str
    category: str
    direction: Direction
    period: Union[FixedPeriod, MonthlyPeriod]
    typical_amount: Money
    estimator: Estimator
    last_seen: date
    evidence_event_ids: Tuple[str, ...]
    confidence: float
    user_confirmed: bool = False
    description: str = ""
    flexibility: Flexibility = Flexibility.FIXED
    min_allowed_amount: Optional[Money] = None


@dataclass(frozen=True)
class IncomeStream:
    """A recurring income stream (salary, freelance, rent received...)."""

    stream_key: str
    category: str
    description: str
    history: Tuple[FinancialEvent, ...]
    future_confirmed: Tuple[FinancialEvent, ...]
    pattern: Optional[RecurringPattern] = None

    @property
    def typical_amount(self) -> Optional[Money]:
        return self.pattern.typical_amount if self.pattern else None


@dataclass(frozen=True)
class ExpenseStream:
    """A recurring expense stream (rent, subscription, insurance...)."""

    stream_key: str
    category: str
    description: str
    history: Tuple[FinancialEvent, ...]
    future_confirmed: Tuple[FinancialEvent, ...]
    pattern: Optional[RecurringPattern] = None
    flexibility: Flexibility = Flexibility.FIXED
    min_allowed_amount: Optional[Money] = None
    is_protected: bool = False
    can_reduce: bool = False
    can_stop: bool = False

    @property
    def typical_amount(self) -> Optional[Money]:
        return self.pattern.typical_amount if self.pattern else None


# ---------------------------------------------------------------------------
# User profile and requests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UserFinancialProfile:
    """User's balances, preferences, and authorizations.

    The category sets are *authorizations*: the planner and verifier may only
    stop/reduce streams whose category the user explicitly opted in to, and
    may never touch protected categories.
    """

    user_id: str
    home_currency: str
    current_available_balance: Money
    minimum_balance_to_keep: Money
    financial_priorities: Tuple[str, ...] = ()
    protected_categories: frozenset = frozenset()
    reducible_categories: frozenset = frozenset()
    stoppable_categories: frozenset = frozenset()
    payment_methods_considered: frozenset = frozenset({PaymentMethod.FULL_PAYMENT})
    max_installment_payments: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("current_available_balance", "minimum_balance_to_keep"):
            value: Money = getattr(self, name)
            if value.currency != self.home_currency:
                raise CurrencyMismatchError(
                    f"{name} currency {value.currency} != home currency {self.home_currency}"
                )
        if self.minimum_balance_to_keep.is_negative:
            raise EngineError("minimum_balance_to_keep cannot be negative")
        if not self.payment_methods_considered:
            raise EngineError("profile must consider at least one payment method")
        for category_set in ("protected_categories", "reducible_categories",
                             "stoppable_categories"):
            if not isinstance(getattr(self, category_set), frozenset):
                raise EngineError(f"{category_set} must be a frozenset")
        overlapping = self.protected_categories & (
            self.reducible_categories | self.stoppable_categories
        )
        if overlapping:
            raise EngineError(
                f"categories cannot be both protected and flexible: {sorted(overlapping)}"
            )


@dataclass(frozen=True)
class PaymentOption:
    """A user-supplied installment option for a specific request."""

    option_id: str
    first_payment_date: date
    frequency_days: int
    number_of_payments: int
    payment_amount: Money
    total_payable: Money

    def __post_init__(self) -> None:
        if self.number_of_payments < 1:
            raise InvalidRequestError("number_of_payments must be >= 1")
        if self.frequency_days < 1:
            raise InvalidRequestError("frequency_days must be >= 1")
        if not self.payment_amount.is_positive:
            raise InvalidRequestError("payment_amount must be positive")

    def build_payments(self) -> Tuple["Payment", ...]:
        """Expand the option into its concrete payment schedule."""
        payments = []
        current = self.first_payment_date
        for _ in range(self.number_of_payments):
            payments.append(Payment(current, self.payment_amount))
            current = current + timedelta(days=self.frequency_days)
        return tuple(payments)


@dataclass(frozen=True)
class AffordabilityRequest:
    """The user's question: can I pay ``requested_amount`` by ``desired_completion_date``?"""

    request_id: str
    requested_amount: Money
    request_date: date
    desired_completion_date: date
    allows_partial_payment: bool = False
    payment_options: Tuple[PaymentOption, ...] = ()

    def __post_init__(self) -> None:
        if not self.requested_amount.is_positive:
            raise InvalidRequestError("requested_amount must be positive")
        if self.desired_completion_date < self.request_date:
            raise InvalidRequestError(
                "desired_completion_date cannot be before request_date"
            )
        for option in self.payment_options:
            if option.payment_amount.currency != self.requested_amount.currency:
                raise CurrencyMismatchError(
                    f"option {option.option_id} currency differs from request currency"
                )


# ---------------------------------------------------------------------------
# Plans and simulation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Payment:
    """A single planned payment: an amount on a date."""

    date: date
    amount: Money


@dataclass(frozen=True)
class SpendingChange:
    """A change to a recurring expense stream that a plan relies on."""

    action: SpendingChangeAction
    pattern_id: str
    per_occurrence_relief: Money
    new_amount: Optional[Money] = None  # required for REDUCE_TO


@dataclass(frozen=True)
class SimulationScenario:
    """What a simulation adds on top of the canonical state.

    The forecast is a pure function of (state, assumptions, scenario); the
    verifier re-simulates a plan simply by passing ``plan.as_scenario()`` —
    it never trusts the planner's own arithmetic.
    """

    payments: Tuple[Payment, ...] = ()
    spending_changes: Tuple[SpendingChange, ...] = ()


@dataclass(frozen=True)
class PaymentPlan:
    """An explicit candidate strategy for satisfying a request."""

    plan_id: str
    method: PaymentMethod
    payments: Tuple[Payment, ...]
    total_payable: Money
    spending_changes: Tuple[SpendingChange, ...] = ()
    option_id: Optional[str] = None

    @property
    def first_payment_date(self) -> Optional[date]:
        return self.payments[0].date if self.payments else None

    @property
    def last_payment_date(self) -> Optional[date]:
        return self.payments[-1].date if self.payments else None

    def as_scenario(self) -> SimulationScenario:
        """The forecast scenario that simulates exactly this plan."""
        return SimulationScenario(payments=self.payments,
                                  spending_changes=self.spending_changes)


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CashFlow:
    """One flow inside a forecast day, with a reference to its cause.

    The ``ref`` prefix explains *why* the balance moved: ``event:<id>`` (a
    ledger event), ``pending:<id>`` (a reserved pending debit),
    ``forecast:<pattern_id>:<date>`` (a generated pattern occurrence), or
    ``payment:<n>`` (a simulated plan payment).
    """

    ref: str
    direction: Direction
    amount: Money
    description: str


@dataclass(frozen=True)
class ForecastPoint:
    """The balance for one day, with the flows that changed it."""

    date: date
    opening_balance: Money
    closing_balance: Money
    flows: Tuple[CashFlow, ...]

    @property
    def net_flow(self) -> Money:
        total = Money.zero(self.closing_balance.currency)
        for flow in self.flows:
            if flow.direction is Direction.CREDIT:
                total = total + flow.amount
            else:
                total = total - flow.amount
        return total


@dataclass(frozen=True)
class CashflowSeries:
    """A daily balance forecast: one point per day over the horizon.

    The series is complete — every calendar day from ``start_date`` through
    ``start_date + horizon_days`` (inclusive) has a point — so any
    transaction or payment date in range has a balance.
    """

    start_date: date
    horizon_days: int
    points: Tuple[ForecastPoint, ...]
    engine_version: str
    input_hash: str

    def balance_on(self, day: date) -> Money:
        """Closing balance on ``day``. Raises KeyError if outside the series."""
        for point in self.points:
            if point.date == day:
                return point.closing_balance
        raise KeyError(f"{day.isoformat()} is outside the forecast series")

    def min_point(self) -> ForecastPoint:
        """The lowest-balance day (earliest date on ties). Deterministic."""
        return min(self.points, key=lambda p: (p.closing_balance.amount_minor, p.date))


@dataclass(frozen=True)
class Assumptions:
    """Named, versioned forecasting assumptions — never ambient globals."""

    horizon_days: int = 90
    #: Pending credits are never counted as available money (hard rule);
    #: the flag exists so tests can document the default and future
    #: product decisions can revisit it explicitly.
    count_pending_credits: bool = False


# ---------------------------------------------------------------------------
# Evidence, verification, recommendation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Evidence:
    """A structured, checkable fact supporting a decision.

    The future LLM explanation layer is *constrained* to these facts: if a
    generated explanation mentions a number not present in the evidence
    bundle, that is a detectable policy violation.
    """

    evidence_type: EvidenceType
    source: str
    fact: str
    date: Optional[date] = None
    amount: Optional[Money] = None
    confidence: Optional[float] = None
    impact: Impact = Impact.SUPPORTING


@dataclass(frozen=True)
class Violation:
    """A machine-readable reason a plan failed the verification gate."""

    code: ViolationCode
    message: str
    date: Optional[date] = None
    amount: Optional[Money] = None


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of the independent verification gate on one plan."""

    plan_id: str
    passed: bool
    violations: Tuple[Violation, ...]
    evidence: Tuple[Evidence, ...]
    engine_version: str
    input_hash: str


@dataclass(frozen=True)
class Recommendation:
    """The final output of analyzing a request: a verified, ranked plan.

    Invariant: ``plan.verification.passed`` is True for every plan that
    reaches a user (enforced upstream in the API serializer; guaranteed here
    by construction — ``analyze_request`` only selects verified plans).
    """

    request_id: str
    status: AffordabilityStatus
    plan: PaymentPlan
    verification: VerificationResult
    ranked_alternatives: Tuple[PaymentPlan, ...]
    evidence: Tuple[Evidence, ...]
    engine_version: str
    input_hash: str


# ---------------------------------------------------------------------------
# Financial state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConflictResolution:
    """A logged resolution of two events claiming the same future slot."""

    stream_key: str
    date: date
    direction: Direction
    kept_event_id: str
    dropped_event_ids: Tuple[str, ...]
    rule: str  # "explicit_amendment" | "status_precedence" | "financially_safer"


@dataclass(frozen=True)
class FinancialState:
    """A user's complete financial situation as of ``as_of``.

    Built exclusively by :func:`app.core.state.build_state`, which enforces
    the classification and conflict-resolution rules. All money in the state
    is in ``home_currency`` — the builder refuses mixed currencies.
    """

    user_id: str
    as_of: date
    home_currency: str
    current_balance: Money
    profile: UserFinancialProfile
    history: Tuple[FinancialEvent, ...]          # settled events before as_of
    future_confirmed: Tuple[FinancialEvent, ...]  # scheduled events at/after as_of
    pending_debits: Tuple[FinancialEvent, ...]    # reserved at forecast start
    pending_credits: Tuple[FinancialEvent, ...]   # tracked, never counted
    patterns: Tuple[RecurringPattern, ...]
    income_streams: Tuple[IncomeStream, ...]
    expense_streams: Tuple[ExpenseStream, ...]
    conflict_log: Tuple[ConflictResolution, ...]

    # -- preference passthroughs (single source of truth: profile) -----------

    @property
    def minimum_balance(self) -> Money:
        return self.profile.minimum_balance_to_keep

    @property
    def protected_categories(self) -> frozenset:
        return self.profile.protected_categories

    @property
    def reducible_categories(self) -> frozenset:
        return self.profile.reducible_categories

    @property
    def stoppable_categories(self) -> frozenset:
        return self.profile.stoppable_categories

    @property
    def payment_methods_considered(self) -> frozenset:
        return self.profile.payment_methods_considered

    @property
    def financial_priorities(self) -> Tuple[str, ...]:
        return self.profile.financial_priorities

    @property
    def confirmed_future_income(self) -> Tuple[FinancialEvent, ...]:
        return tuple(e for e in self.future_confirmed
                     if e.direction is Direction.CREDIT)

    @property
    def confirmed_future_payments(self) -> Tuple[FinancialEvent, ...]:
        return tuple(e for e in self.future_confirmed
                     if e.direction is Direction.DEBIT)

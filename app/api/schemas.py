"""Pydantic schemas for the REST API."""
import datetime
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Common / Core translations
# ---------------------------------------------------------------------------

class MoneySchema(BaseModel):
    amount_minor: int
    currency: str = Field(min_length=3, max_length=3)


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

class UserCreate(BaseModel):
    email: str
    display_name: str = ""
    home_currency: str = "USD"
    current_balance_minor: int = 0
    minimum_balance_minor: int = 0


class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str
    home_currency: str
    created_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True)


class ProfileResponse(BaseModel):
    revision: int
    current_balance_minor: int
    minimum_balance_minor: int
    currency: str
    protected_categories: list[str]
    payment_methods_considered: list[str]

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------

class TransactionCreate(BaseModel):
    date: datetime.date
    direction: str = Field(pattern="^(credit|debit)$")
    amount_minor: int
    category: str
    description: str = ""
    status: str = "settled"
    flexibility: str = "fixed"
    source_type: str = "manual"
    source_id: str = "api"


class TransactionResponse(BaseModel):
    id: str
    date: datetime.date
    direction: str
    amount_minor: int
    currency: str
    category: str
    description: str
    status: str
    source_type: str

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Affordability & Decisions
# ---------------------------------------------------------------------------

class PaymentOptionSchema(BaseModel):
    option_id: str
    first_payment_date: datetime.date
    frequency_days: int
    number_of_payments: int
    payment_amount_minor: int
    total_payable_minor: int


class AffordabilityRequestSchema(BaseModel):
    requested_amount_minor: int
    desired_completion_date: datetime.date
    allows_partial_payment: bool = False
    payment_options: list[PaymentOptionSchema] = Field(default_factory=list)


class DecisionResponse(BaseModel):
    request_id: str
    status: str
    plan_method: str
    total_payable_minor: int
    engine_version: str
    fact_sheet: list[dict]

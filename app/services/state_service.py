"""Deprecated: split into ``financial_state_service`` and ``forecast_service``.

This shim keeps old imports working; new code should import from the
split modules directly.
"""
from .financial_state_service import get_financial_state  # noqa: F401
from .forecast_service import run_forecast  # noqa: F401

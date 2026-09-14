import unittest
import uuid
from datetime import date
from pydantic import ValidationError
from sqlalchemy import delete

from sqlalchemy.orm import Session
from app.db.models import User, Base
from app.agent.tools import registry, StateTool, ForecastTool, AffordabilityTool, TransactionTool
from app.tests.test_api import ApiTestBase
from app.services.user_service import create_user

class TestTools(ApiTestBase):

    def setUp(self):
        # Create a fresh test user (with a financial-profile revision) for each test
        with self.SessionFactory() as db:
            user = create_user(
                db,
                email=f"{uuid.uuid4().hex[:10]}@example.com",
                home_currency="USD",
                current_balance_minor=1_000_000,
            )
            db.commit()
            self.test_user_id = user.id

    def tearDown(self):
        # Remove the user and every child row referencing it so each test
        # starts from a clean slate in the shared class-level in-memory DB.
        with self.SessionFactory() as db:
            for table in reversed(Base.metadata.sorted_tables):
                if "user_id" in table.c:
                    db.execute(delete(table).where(table.c.user_id == self.test_user_id))
            db.execute(delete(User).where(User.id == self.test_user_id))
            db.commit()

    def _get_user(self, db: Session) -> User:
        return db.get(User, self.test_user_id)

    def test_registry_schemas(self):
        """Verify that all tools in the registry export valid JSON schemas."""
        schemas = registry.get_all_schemas()
        names = {s["name"] for s in schemas}

        self.assertIn("get_financial_state", names)
        self.assertIn("run_forecast", names)
        self.assertIn("check_affordability", names)
        self.assertIn("add_transaction", names)

        # Basic shape validation
        state_schema = next(s for s in schemas if s["name"] == "get_financial_state")
        self.assertIn("input_schema", state_schema)
        self.assertTrue(state_schema["description"])

    def test_tool_retrieval(self):
        """Verify tool retrieval works and raises KeyError for unknown tools."""
        tool = registry.get_tool("get_financial_state")
        self.assertIsInstance(tool, StateTool)

        with self.assertRaisesRegex(KeyError, "Tool 'unknown_tool' not found in registry."):
            registry.get_tool("unknown_tool")

    def test_tool_validation_dispatch(self):
        """Test that arguments are validated via Pydantic before tool execution."""
        as_of = date(2026, 9, 14)

        with self.SessionFactory() as db:
            user = self._get_user(db)
            # Valid arguments dispatch
            result = registry.execute(
                "get_financial_state",
                db=db,
                user=user,
                args={},
                as_of=as_of
            )

            self.assertEqual(result["user_id"], user.id)
            self.assertEqual(result["home_currency"], user.home_currency)

            # Invalid arguments should raise a validation error
            with self.assertRaises(ValidationError):
                registry.execute(
                    "add_transaction",
                    db=db,
                    user=user,
                    args={
                        "date": "not-a-date",  # Bad date
                        "direction": "debit",
                        "amount_minor": 1000,
                        "category": "Salary",
                    },
                    as_of=as_of
                )

    def test_transaction_tool_execution(self):
        """Test that the transaction tool correctly handles execution and domain errors."""
        as_of = date(2026, 9, 14)

        args = {
            "date": "2026-09-15",
            "direction": "credit",
            "amount_minor": 5000,
            "category": "Bonus",
            "description": "Yearly bonus",
            "status": "settled"
        }

        with self.SessionFactory() as db:
            user = self._get_user(db)
            result = registry.execute(
                "add_transaction",
                db=db,
                user=user,
                args=args,
                as_of=as_of
            )

            self.assertTrue(result["success"])
            self.assertIn("transaction_id", result)

            # Try adding the exact identical transaction again to trigger a dedup error
            result_dup = registry.execute(
                "add_transaction",
                db=db,
                user=user,
                args=args,
                as_of=as_of
            )

            self.assertFalse(result_dup["success"])
            self.assertEqual(result_dup["error_code"], "duplicate")
            self.assertIn("an identical transaction already exists", result_dup["message"])

    def test_forecast_tool_execution(self):
        """Test generating a forecast via the tool."""
        as_of = date(2026, 9, 14)
        args = {
            "horizon_days": 30,
            "scenario_payments": [
                {
                    "date": "2026-09-20",
                    "amount_minor": 2000
                }
            ]
        }

        with self.SessionFactory() as db:
            user = self._get_user(db)
            result = registry.execute(
                "run_forecast",
                db=db,
                user=user,
                args=args,
                as_of=as_of
            )

            self.assertEqual(result["horizon_days"], 30)
            self.assertGreater(len(result["points"]), 0)

    def test_affordability_tool_execution(self):
        """Test affordability check execution."""
        as_of = date(2026, 9, 14)
        args = {
            "requested_amount_minor": 1500,
            "desired_completion_date": "2026-09-21",
            "allows_partial_payment": False,
            "horizon_days": 90
        }

        with self.SessionFactory() as db:
            user = self._get_user(db)
            result = registry.execute(
                "check_affordability",
                db=db,
                user=user,
                args=args,
                as_of=as_of
            )

            self.assertIn("status", result)
            self.assertIn("fact_sheet", result)

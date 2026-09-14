import unittest
import uuid
import json
from datetime import date
from typing import List

from pydantic import ValidationError

from sqlalchemy import delete
from sqlalchemy.orm import Session
from app.db.models import User, Base
from app.tests.test_api import ApiTestBase
from app.services.user_service import create_user

from app.agent.orchestrator import (
    Role, Message, ToolCall, LLMResponse, LLMProvider,
    AgentOrchestrator, MaxStepsExceededError
)

class CannedLLM(LLMProvider):
    """A deterministic mock LLM returning canned responses for tests."""

    def __init__(self, responses: List[LLMResponse]):
        self.responses = responses
        self.call_count = 0
        self.history_snapshots = []

    def generate(self, messages: List[Message]) -> LLMResponse:
        # Snapshot history to allow assertions post-execution
        self.history_snapshots.append([m.model_copy() for m in messages])
        if self.call_count >= len(self.responses):
            raise RuntimeError("CannedLLM ran out of responses")
        resp = self.responses[self.call_count]
        self.call_count += 1
        return resp


class TestAgentOrchestrator(ApiTestBase):

    def setUp(self):
        with self.SessionFactory() as db:
            self.user = create_user(
                db,
                email=f"{uuid.uuid4().hex[:10]}@example.com",
                home_currency="USD",
                current_balance_minor=10000,
            )
            db.commit()
            self.test_user_id = self.user.id
            self.today = date(2026, 9, 14)

    def tearDown(self):
        with self.SessionFactory() as db:
            for table in reversed(Base.metadata.sorted_tables):
                if "user_id" in table.c:
                    db.execute(delete(table).where(table.c.user_id == self.test_user_id))
            db.execute(delete(User).where(User.id == self.test_user_id))
            db.commit()

    def test_single_tool_execution(self):
        """Test standard tool call then final answer."""
        llm = CannedLLM([
            LLMResponse(
                text_content=None,
                tool_calls=[ToolCall(id="call_1", name="get_financial_state", arguments={})]
            ),
            LLMResponse(
                text_content="Your balance is $100.00.",
                tool_calls=None
            )
        ])

        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            orchestrator = AgentOrchestrator(db, user, llm, max_steps=5)
            final = orchestrator.run("What is my balance?", self.today)

            self.assertEqual(final, "Your balance is $100.00.")
            self.assertEqual(llm.call_count, 2)

            # Check message logic
            # System (0), User (0), ...
            messages = orchestrator.messages
            self.assertEqual(messages[0].role, Role.USER)
            self.assertEqual(messages[1].role, Role.ASSISTANT)
            self.assertEqual(messages[1].tool_calls[0].name, "get_financial_state")
            self.assertEqual(messages[2].role, Role.TOOL)
            self.assertFalse(messages[2].is_error)

            # Ensure the tool result has actual financial state data
            result_data = json.loads(messages[2].content)
            self.assertEqual(result_data["home_currency"], "USD")
            self.assertEqual(result_data["current_balance_minor"], 10000)

    def test_multiple_sequential_tools(self):
        """Test chain of tool calls mimicking a typical reasoning sequence."""
        llm = CannedLLM([
            # Step 1: Check state
            LLMResponse(
                tool_calls=[ToolCall(id="call_1", name="get_financial_state", arguments={})]
            ),
            # Step 2: Run a forecast with the context provided
            LLMResponse(
                tool_calls=[ToolCall(id="call_2", name="run_forecast", arguments={"horizon_days": 30})]
            ),
            # Step 3: Final answer
            LLMResponse(text_content="Done forecasting.")
        ])

        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            orchestrator = AgentOrchestrator(db, user, llm, max_steps=5)
            final = orchestrator.run("Forecast my balance for a month", self.today)

            self.assertEqual(final, "Done forecasting.")
            # Verify proper trace execution order
            roles = [m.role for m in orchestrator.messages]
            self.assertEqual(roles, [Role.USER, Role.ASSISTANT, Role.TOOL, Role.ASSISTANT, Role.TOOL, Role.ASSISTANT])

    def test_invalid_unknown_tool_recovery(self):
        """Test orchestrator captures KeyError correctly providing a chance to recover."""
        llm = CannedLLM([
            # Hallucinate a tool
            LLMResponse(
                tool_calls=[ToolCall(id="call_99", name="hack_mainframe", arguments={})]
            ),
            # Model recovers after seeing the error
            LLMResponse(text_content="Oops, I cannot do that.")
        ])

        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            orchestrator = AgentOrchestrator(db, user, llm)
            final = orchestrator.run("Hack it.", self.today)

            tool_msg = orchestrator.messages[2]
            self.assertEqual(tool_msg.role, Role.TOOL)
            self.assertTrue(tool_msg.is_error)
            self.assertIn("KeyError", tool_msg.content)
            self.assertIn("hack_mainframe", tool_msg.content)

            self.assertEqual(final, "Oops, I cannot do that.")

    def test_invalid_tool_arguments_recovery(self):
        """Test orchestrator captures Pydantic ValidationError into the context."""
        llm = CannedLLM([
            # Provide unparseable types (wrong format constraint check)
            LLMResponse(
                tool_calls=[ToolCall(id="call_tx", name="add_transaction", arguments={
                    "date": "very-wrong-date",
                    "direction": "credit",
                    "amount_minor": 1000,
                    "category": "Salary"
                })]
            ),
            # Recover
            LLMResponse(text_content="Let me reformat that")
        ])

        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            orchestrator = AgentOrchestrator(db, user, llm)
            orchestrator.run("Add salary.", self.today)

            tool_msg = orchestrator.messages[2]
            self.assertTrue(tool_msg.is_error)
            self.assertIn("ValidationError", tool_msg.content)

    def test_tool_execution_failure_recovery(self):
        """Test catching a domain logic error."""
        # add_transaction returns a dict with success=False, but here we can just test if the result itself has success False
        # wait, transaction_tool catches the TransactionServiceError and returns {"success": False}.
        # meaning, it won't throw an exception at the orchestrator layer. We should verify it passes it properly.
        llm = CannedLLM([
            LLMResponse(
                tool_calls=[ToolCall(id="call_dup", name="add_transaction", arguments={
                    "date": "2026-09-14",
                    "direction": "up", # invalid direction enum for domain but matches basic string checks
                    "amount_minor": 1000,
                    "category": "Salary"
                })]
            ),
            LLMResponse(text_content="Got a business logic failure")
        ])

        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            orchestrator = AgentOrchestrator(db, user, llm)
            orchestrator.run("Add something", self.today)

            tool_msg = orchestrator.messages[2]
            self.assertEqual(tool_msg.role, Role.TOOL)
            result = json.loads(tool_msg.content)
            self.assertFalse(result["success"])
            self.assertEqual(result["error_code"], "invalid_enum")

    def test_max_steps_exceeded(self):
        """Test loop terminates via MaxStepsExceededError if LLM continuously loops."""
        # Continually ask to get_financial_state
        llm = CannedLLM([
            LLMResponse(tool_calls=[ToolCall(id=f"c_{i}", name="get_financial_state", arguments={})])
            for i in range(10)
        ] + [LLMResponse(text_content="Never reached")])

        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            orchestrator = AgentOrchestrator(db, user, llm, max_steps=3)
            with self.assertRaises(MaxStepsExceededError):
                orchestrator.run("Loop me", self.today)

            self.assertEqual(llm.call_count, 3)

    def test_tool_database_isolation(self):
        """Test that successfully dispatched modifying tools correctly manipulate DB."""
        llm = CannedLLM([
            LLMResponse(
                tool_calls=[ToolCall(id="c1", name="add_transaction", arguments={
                    "date": "2026-09-13",
                    "direction": "credit",
                    "amount_minor": 55500,
                    "category": "Bonus"
                })]
            ),
            LLMResponse(text_content="Done")
        ])

        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            orchestrator = AgentOrchestrator(db, user, llm)
            orchestrator.run("Add a bonus", self.today)
            db.commit()

            # Verify via standalone DB query that the tool's ledger ingestion persisted in isolated session
            from app.db.models import Transaction, FinancialProfile
            txns = db.query(Transaction).filter_by(user_id=self.test_user_id).all()
            self.assertEqual(len(txns), 1)
            self.assertEqual(txns[0].amount_minor, 55500)
            self.assertEqual(txns[0].direction, "credit")

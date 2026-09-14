"""Canned-transcript end-to-end agent tests (ROADMAP Phase 3, final segment).

These tests exercise the complete Phase 3 chain with a scripted (canned) LLM:

    User request
      -> ContextBuilder (system prompt + state-wrapped user prompt)
      -> AgentOrchestrator loop
      -> ScriptedLLM canned responses
      -> Tool Registry (Pydantic-validated dispatch)
      -> Phase 2 services / Phase 1 deterministic engine
      -> tool result returned into the orchestrator history
      -> subsequent canned LLM response
      -> final assistant response

The ScriptedLLM consumes predefined responses only and performs no network,
API, or filesystem access. No second business-logic implementation exists
here: every financial fact is produced by the real services and engine.
"""
import json
import unittest
import uuid
from datetime import date
from typing import List

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.db.models import User, Base, Transaction
from app.tests.test_api import ApiTestBase
from app.services.user_service import create_user
from app.agent.orchestrator import (
    Role, Message, ToolCall, LLMResponse, LLMProvider,
    AgentOrchestrator, MaxStepsExceededError,
)
from app.agent.context import ContextBuilder
from app.agent.tools.registry import registry


class ScriptedLLM(LLMProvider):
    """A deterministic canned LLM: replays a fixed transcript, records history.

    It implements the same LLMProvider protocol a real adapter will, so these
    tests run the identical orchestration path production will use — minus
    any network call.
    """

    def __init__(self, responses: List[LLMResponse]):
        self.responses = responses
        self.call_count = 0
        self.received: List[List[Message]] = []

    def generate(self, messages: List[Message]) -> LLMResponse:
        self.received.append([m.model_copy() for m in messages])
        if self.call_count >= len(self.responses):
            raise RuntimeError("ScriptedLLM ran out of canned responses")
        resp = self.responses[self.call_count]
        self.call_count += 1
        return resp


class TestAgentEndToEnd(ApiTestBase):
    """Full-chain canned-transcript scenarios."""

    def setUp(self):
        with self.SessionFactory() as db:
            self.user = create_user(
                db,
                email=f"{uuid.uuid4().hex[:10]}@example.com",
                home_currency="USD",
                current_balance_minor=1_000_000,
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

    def _build_context(self, db: Session, request: str):
        """Assemble the exact context an agent run would start from."""
        user = db.get(User, self.test_user_id)
        system_prompt = ContextBuilder.build_system_prompt(user, self.today)
        user_prompt = ContextBuilder.build_user_prompt(db, user, request, self.today)
        return user, system_prompt, user_prompt

    # --- Scenario 1: single tool call, full chain ---

    def test_e2e_single_tool_request(self):
        """One tool call answers a balance question end to end."""
        llm = ScriptedLLM([
            LLMResponse(tool_calls=[
                ToolCall(id="call_1", name="get_financial_state", arguments={})
            ]),
            LLMResponse(text_content="Your current balance is $10,000.00."),
        ])

        with self.SessionFactory() as db:
            user, system_prompt, user_prompt = self._build_context(db, "What is my balance?")
            orchestrator = AgentOrchestrator(db, user, llm, system_prompt=system_prompt)
            final = orchestrator.run(user_prompt, self.today)

            self.assertEqual(final, "Your current balance is $10,000.00.")

            # Full-chain role sequence with context assembly included
            roles = [m.role for m in orchestrator.messages]
            self.assertEqual(roles, [Role.SYSTEM, Role.USER, Role.ASSISTANT, Role.TOOL, Role.ASSISTANT])

            # The tool result carries real engine/service output, not a canned echo
            tool_msg = orchestrator.messages[3]
            self.assertFalse(tool_msg.is_error)
            self.assertEqual(tool_msg.tool_call_id, "call_1")
            state = json.loads(tool_msg.content)
            self.assertEqual(state["user_id"], user.id)
            self.assertEqual(state["home_currency"], "USD")
            self.assertEqual(state["current_balance_minor"], 1_000_000)
            self.assertEqual(state["as_of"], "2026-09-14")

    # --- Scenario 2: multiple sequential tool calls ---

    def test_e2e_multiple_sequential_tools(self):
        """State inspection followed by a forecast, chained through history."""
        llm = ScriptedLLM([
            LLMResponse(tool_calls=[
                ToolCall(id="call_1", name="get_financial_state", arguments={})
            ]),
            LLMResponse(tool_calls=[
                ToolCall(id="call_2", name="run_forecast", arguments={"horizon_days": 30})
            ]),
            LLMResponse(text_content="Forecast complete: your balance stays positive for 30 days."),
        ])

        with self.SessionFactory() as db:
            user, system_prompt, user_prompt = self._build_context(
                db, "Check my state, then forecast the next month."
            )
            orchestrator = AgentOrchestrator(db, user, llm, system_prompt=system_prompt)
            final = orchestrator.run(user_prompt, self.today)

            self.assertEqual(final, "Forecast complete: your balance stays positive for 30 days.")
            roles = [m.role for m in orchestrator.messages]
            self.assertEqual(roles, [
                Role.SYSTEM, Role.USER,
                Role.ASSISTANT, Role.TOOL,
                Role.ASSISTANT, Role.TOOL,
                Role.ASSISTANT,
            ])

            # Second tool result is the forecast, produced by the real engine
            forecast_msg = orchestrator.messages[5]
            self.assertFalse(forecast_msg.is_error)
            self.assertEqual(forecast_msg.tool_call_id, "call_2")
            forecast = json.loads(forecast_msg.content)
            self.assertEqual(forecast["horizon_days"], 30)
            self.assertGreater(len(forecast["points"]), 0)

    # --- Scenario 3: invalid/unknown tool recovery ---

    def test_e2e_unknown_tool_recovery(self):
        """A hallucinated tool name becomes an error the agent recovers from."""
        llm = ScriptedLLM([
            LLMResponse(tool_calls=[
                ToolCall(id="call_99", name="wire_money_overseas", arguments={})
            ]),
            LLMResponse(text_content="I cannot do that — no such tool is available."),
        ])

        with self.SessionFactory() as db:
            user, system_prompt, user_prompt = self._build_context(db, "Send all my money abroad.")
            orchestrator = AgentOrchestrator(db, user, llm, system_prompt=system_prompt)
            final = orchestrator.run(user_prompt, self.today)

            self.assertEqual(final, "I cannot do that — no such tool is available.")
            tool_msg = orchestrator.messages[3]
            self.assertEqual(tool_msg.role, Role.TOOL)
            self.assertTrue(tool_msg.is_error)
            self.assertIn("wire_money_overseas", tool_msg.content)

    # --- Scenario 4: invalid tool arguments / schema validation recovery ---

    def test_e2e_invalid_arguments_recovery(self):
        """Bad arguments fail Pydantic validation and the agent recovers."""
        llm = ScriptedLLM([
            LLMResponse(tool_calls=[
                ToolCall(id="call_tx", name="add_transaction", arguments={
                    "date": "not-a-real-date",
                    "direction": "credit",
                    "amount_minor": 5000,
                    "category": "Salary",
                })
            ]),
            LLMResponse(text_content="That date was invalid; I will not add the transaction."),
        ])

        with self.SessionFactory() as db:
            user, system_prompt, user_prompt = self._build_context(db, "Add my salary.")
            orchestrator = AgentOrchestrator(db, user, llm, system_prompt=system_prompt)
            final = orchestrator.run(user_prompt, self.today)

            self.assertIn("will not add", final)
            tool_msg = orchestrator.messages[3]
            self.assertTrue(tool_msg.is_error)

            # The invalid write must NOT have persisted anything
            db.commit()
            txns = db.query(Transaction).filter_by(user_id=self.test_user_id).all()
            self.assertEqual(len(txns), 0)

    # --- Scenario 5: tool execution / domain failure recovery ---

    def test_e2e_domain_failure_recovery(self):
        """A domain rejection flows back as a structured failure, not an exception."""
        llm = ScriptedLLM([
            LLMResponse(tool_calls=[
                ToolCall(id="call_bad", name="add_transaction", arguments={
                    "date": "2026-09-14",
                    "direction": "sideways",  # passes the str schema, rejected by the service
                    "amount_minor": 1000,
                    "category": "Salary",
                })
            ]),
            LLMResponse(text_content="The transaction was rejected because the direction is invalid."),
        ])

        with self.SessionFactory() as db:
            user, system_prompt, user_prompt = self._build_context(db, "Add something weird.")
            orchestrator = AgentOrchestrator(db, user, llm, system_prompt=system_prompt)
            final = orchestrator.run(user_prompt, self.today)

            self.assertIn("rejected", final)
            tool_msg = orchestrator.messages[3]
            self.assertEqual(tool_msg.role, Role.TOOL)
            # Domain failures are tool *results* (success: False), not exceptions
            self.assertFalse(tool_msg.is_error)
            result = json.loads(tool_msg.content)
            self.assertFalse(result["success"])
            self.assertEqual(result["error_code"], "invalid_enum")

    # --- Scenario 6: maximum-step / loop-budget protection ---

    def test_e2e_max_steps_protection(self):
        """A tool-call loop is cut off by the step budget."""
        llm = ScriptedLLM(
            [LLMResponse(tool_calls=[
                ToolCall(id=f"c_{i}", name="get_financial_state", arguments={})
            ]) for i in range(10)]
        )

        with self.SessionFactory() as db:
            user, system_prompt, user_prompt = self._build_context(db, "Loop forever.")
            orchestrator = AgentOrchestrator(db, user, llm, system_prompt=system_prompt, max_steps=3)
            with self.assertRaises(MaxStepsExceededError):
                orchestrator.run(user_prompt, self.today)

            # The budget, not the transcript length, ended the run
            self.assertEqual(llm.call_count, 3)

    # --- Scenario 7: state-changing tool call with persistence verification ---

    def test_e2e_state_changing_tool_persists(self):
        """add_transaction through the agent loop lands in the database."""
        llm = ScriptedLLM([
            LLMResponse(tool_calls=[
                ToolCall(id="call_add", name="add_transaction", arguments={
                    "date": "2026-09-13",
                    "direction": "credit",
                    "amount_minor": 250_000,
                    "category": "Salary",
                    "description": "September salary",
                })
            ]),
            LLMResponse(text_content="Your salary transaction has been added."),
        ])

        with self.SessionFactory() as db:
            user, system_prompt, user_prompt = self._build_context(db, "Add my September salary.")
            orchestrator = AgentOrchestrator(db, user, llm, system_prompt=system_prompt)
            final = orchestrator.run(user_prompt, self.today)
            self.assertEqual(final, "Your salary transaction has been added.")

            # The tool reported success with a real transaction id
            tool_result = json.loads(orchestrator.messages[3].content)
            self.assertTrue(tool_result["success"])
            self.assertIn("transaction_id", tool_result)

            db.commit()
            txns = db.query(Transaction).filter_by(user_id=self.test_user_id).all()
            self.assertEqual(len(txns), 1)
            self.assertEqual(txns[0].amount_minor, 250_000)
            self.assertEqual(txns[0].direction, "credit")
            self.assertEqual(txns[0].category, "Salary")

            # The persisted row flows back through the tool boundary: a fresh
            # state fetch (same registry path the agent would use) sees it.
            state_result = registry.execute(
                "get_financial_state", db=db, user=user, args={}, as_of=self.today
            )
            self.assertEqual(state_result["counts"]["history_events"], 1)

    # --- Scenario 8: final response after successful tool execution ---

    def test_e2e_affordability_final_response(self):
        """A verified affordability answer reaches the final assistant turn."""
        llm = ScriptedLLM([
            LLMResponse(tool_calls=[
                ToolCall(id="call_aff", name="check_affordability", arguments={
                    "requested_amount_minor": 150_000,
                    "desired_completion_date": "2026-09-21",
                })
            ]),
            LLMResponse(
                text_content="Yes — this is affordable under the recommended plan."
            ),
        ])

        with self.SessionFactory() as db:
            user, system_prompt, user_prompt = self._build_context(
                db, "Can I afford a $1,500 purchase next week?"
            )
            orchestrator = AgentOrchestrator(db, user, llm, system_prompt=system_prompt)
            final = orchestrator.run(user_prompt, self.today)

            self.assertEqual(final, "Yes — this is affordable under the recommended plan.")

            # The engine's verified decision is what the agent answered from
            tool_msg = orchestrator.messages[3]
            self.assertFalse(tool_msg.is_error)
            result = json.loads(tool_msg.content)
            self.assertEqual(result["status"], "affordable_now")
            self.assertEqual(result["plan_method"], "full_payment")
            self.assertEqual(result["total_payable_minor"], 150_000)
            self.assertTrue(result["fact_sheet"])

    # --- Scenario 9: context + orchestrator compatibility, result propagation ---

    def test_e2e_context_and_result_propagation(self):
        """The LLM sees the assembled context, then the tool result, verbatim."""
        llm = ScriptedLLM([
            LLMResponse(tool_calls=[
                ToolCall(id="call_1", name="get_financial_state", arguments={})
            ]),
            LLMResponse(text_content="All done."),
        ])

        with self.SessionFactory() as db:
            user, system_prompt, user_prompt = self._build_context(db, "Summarize my finances.")
            orchestrator = AgentOrchestrator(db, user, llm, system_prompt=system_prompt)
            orchestrator.run(user_prompt, self.today)

            # First generate() call: context assembly output, nothing else
            first = llm.received[0]
            self.assertEqual([m.role for m in first], [Role.SYSTEM, Role.USER])
            self.assertIn("AI Financial Decision Agent", first[0].content)
            self.assertIn("get_financial_state", first[0].content)   # tool schemas exposed
            self.assertIn("run_forecast", first[0].content)
            self.assertIn("Today's Date: 2026-09-14", first[0].content)  # explicit as-of
            self.assertIn("Summarize my finances.", first[1].content)
            self.assertIn("CURRENT FINANCIAL STATE SNAPSHOT", first[1].content)

            # Second generate() call: history grew with the assistant tool call
            # and the tool result, correctly linked by tool_call_id
            second = llm.received[1]
            self.assertEqual([m.role for m in second],
                             [Role.SYSTEM, Role.USER, Role.ASSISTANT, Role.TOOL])
            assistant, tool = second[2], second[3]
            self.assertEqual(assistant.tool_calls[0].name, "get_financial_state")
            self.assertEqual(tool.tool_call_id, assistant.tool_calls[0].id)
            self.assertFalse(tool.is_error)
            self.assertIn('"current_balance_minor": 1000000', tool.content)

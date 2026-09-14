import json
import unittest
import uuid
from datetime import date
from typing import List

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.db.models import User, Base
from app.tests.test_api import ApiTestBase
from app.services.user_service import create_user
from app.agent.orchestrator import Role, Message, LLMResponse, LLMProvider, AgentOrchestrator
from app.agent.context import ContextBuilder
from app.agent.tools.registry import registry


class CapturingLLM(LLMProvider):
    """Records the messages it receives and returns a fixed final answer."""

    def __init__(self):
        self.received: List[List[Message]] = []

    def generate(self, messages: List[Message]) -> LLMResponse:
        self.received.append([m.model_copy() for m in messages])
        return LLMResponse(text_content="Final canned answer.")


class TestContextAssembly(ApiTestBase):

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

    # --- System prompt construction ---

    def test_system_prompt_construction(self):
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            prompt = ContextBuilder.build_system_prompt(user, self.today)

            self.assertIsInstance(prompt, str)
            self.assertIn("AI Financial Decision Agent", prompt)
            # Deterministic environment block
            self.assertIn("Today's Date: 2026-09-14", prompt)
            self.assertIn(f"User ID: {user.id}", prompt)
            self.assertIn("Home Currency: USD", prompt)

    def test_system_prompt_includes_tool_definitions(self):
        """All registry tool schemas must be exposed to the LLM."""
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            prompt = ContextBuilder.build_system_prompt(user, self.today)

            for schema in registry.get_all_schemas():
                self.assertIn(schema["name"], prompt)
                self.assertIn(schema["description"], prompt)

            # The schema payload is embedded as JSON, so it must round-trip
            embedded = prompt[prompt.index("AVAILABLE TOOLS"):]
            self.assertIn('"input_schema"', embedded)

    def test_system_prompt_contains_response_constraints(self):
        """Output constraints and anti-hallucination rules must be present."""
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            prompt = ContextBuilder.build_system_prompt(user, self.today)

            self.assertIn("MUST NOT invent", prompt)
            self.assertIn("RESPONSE CONSTRAINTS", prompt)
            self.assertIn("plan_method", prompt)
            self.assertIn("fact_sheet", prompt)
            self.assertIn("success: False", prompt)

    def test_system_prompt_has_no_financial_calculations(self):
        """Context assembly must not embed balances or computed figures."""
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            prompt = ContextBuilder.build_system_prompt(user, self.today)

            self.assertNotIn("10000", prompt)
            self.assertNotIn('"current_balance_minor":', prompt.split("AVAILABLE TOOLS")[0])

    # --- User prompt / financial state context ---

    def test_user_prompt_includes_financial_state(self):
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            user_prompt = ContextBuilder.build_user_prompt(
                db, user, "Can I afford a new laptop?", self.today
            )

            self.assertIn("Can I afford a new laptop?", user_prompt)
            # The state snapshot comes from the existing service boundary
            state = json.loads(
                user_prompt[user_prompt.index("{"):user_prompt.rindex("}") + 1]
            )
            self.assertEqual(state["user_id"], user.id)
            self.assertEqual(state["home_currency"], "USD")
            self.assertEqual(state["current_balance_minor"], 10000)
            self.assertEqual(state["as_of"], "2026-09-14")

    def test_user_prompt_as_of_date_handling(self):
        """The as-of date flows deterministically into the state snapshot."""
        other_day = date(2026, 10, 1)
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            user_prompt = ContextBuilder.build_user_prompt(db, user, "Check state", other_day)
            self.assertIn('"as_of": "2026-10-01"', user_prompt)

    def test_user_prompt_handles_missing_profile(self):
        """A user with no financial profile must still get a usable prompt."""
        with self.SessionFactory() as db:
            orphan_id = f"orphan-{uuid.uuid4().hex[:10]}"
            orphan = User(id=orphan_id, email=f"{orphan_id}@example.com", home_currency="USD")
            db.add(orphan)
            db.commit()
            try:
                user_prompt = ContextBuilder.build_user_prompt(
                    db, orphan, "What is my balance?", self.today
                )
                self.assertIn("What is my balance?", user_prompt)
                self.assertIn("Error retrieving financial state", user_prompt)
            finally:
                db.execute(delete(User).where(User.id == orphan_id))
                db.commit()

    # --- Determinism ---

    def test_context_generation_is_deterministic(self):
        """Repeated builds produce byte-identical prompts for identical inputs."""
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            p1 = ContextBuilder.build_system_prompt(user, self.today)
            p2 = ContextBuilder.build_system_prompt(user, self.today)
            self.assertEqual(p1, p2)

            u1 = ContextBuilder.build_user_prompt(db, user, "Same question", self.today)
            u2 = ContextBuilder.build_user_prompt(db, user, "Same question", self.today)
            self.assertEqual(u1, u2)

    # --- Orchestrator compatibility ---

    def test_orchestrator_compatibility(self):
        """Built context must drive a full orchestrator run unchanged."""
        llm = CapturingLLM()
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            system_prompt = ContextBuilder.build_system_prompt(user, self.today)
            user_prompt = ContextBuilder.build_user_prompt(db, user, "What is my balance?", self.today)

            orchestrator = AgentOrchestrator(db, user, llm, system_prompt=system_prompt)
            final = orchestrator.run(user_prompt, self.today)

            self.assertEqual(final, "Final canned answer.")
            messages = orchestrator.messages
            self.assertEqual(messages[0].role, Role.SYSTEM)
            self.assertEqual(messages[0].content, system_prompt)
            self.assertEqual(messages[1].role, Role.USER)
            self.assertEqual(messages[1].content, user_prompt)
            self.assertEqual(messages[2].role, Role.ASSISTANT)

            # The LLM actually saw the assembled system context
            seen_system = llm.received[0][0]
            self.assertEqual(seen_system.role, Role.SYSTEM)
            self.assertIn("AI Financial Decision Agent", seen_system.content)

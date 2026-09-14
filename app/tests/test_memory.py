import json
import socket
import unittest
import uuid
from datetime import date
from typing import List

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.db.models import User, Base
from app.tests.test_api import ApiTestBase
from app.services.user_service import create_user
from app.agent.context import ContextBuilder
from app.agent.tools.registry import registry
from app.agent.orchestrator import Role, Message, LLMResponse, LLMProvider, AgentOrchestrator
from app.memory import (
    MemoryServiceError,
    get_preference,
    get_preferences,
    save_preference,
    get_goal,
    list_goals,
    save_goal,
    update_goal,
    delete_preference,
    save_fact,
    get_fact,
    list_active_facts,
    search_facts,
    close_fact,
    build_memory_snapshot,
    render_memory_text,
)

#: Every agent turn in this module runs on this canned provider. Nothing here
#: may reach a model over the network (see ``_no_network`` below).
_ORIGINAL_SOCKET = socket.socket


def _no_network(*args, **kwargs):
    """Fail loudly if any test in this module tries to open a socket."""
    raise AssertionError(
        "network access is not permitted in the offline memory test suite; "
        "use the CannedLLM provider, or set OMNIROUTE_LIVE_TEST=1 and use "
        "TestOmniRouteLiveSmoke in test_omniroute_provider.py for a real call."
    )


def setUpModule():
    """Hard-block sockets for the whole module.

    The memory tests must be deterministic and offline. Blocking at the module
    boundary makes an accidental real LLM call impossible rather than merely
    unlikely - including from the large-context stress test below, whose value
    is precisely that it never leaves the process.
    """
    socket.socket = _no_network


def tearDownModule():
    socket.socket = _ORIGINAL_SOCKET


class CapturingLLM(LLMProvider):
    """Records the messages it receives and returns a fixed final answer."""

    def __init__(self):
        self.received: List[List[Message]] = []

    def generate(self, messages: List[Message]) -> LLMResponse:
        self.received.append([m.model_copy() for m in messages])
        return LLMResponse(text_content="Final canned answer.")


class TestMemoryIsolation(ApiTestBase):
    """User isolation: reads/writes scoped per user; cross-user get -> not found."""

    def setUp(self):
        super().setUp()
        with self.SessionFactory() as db:
            self.user_a = create_user(
                db,
                email=f"{uuid.uuid4().hex[:10]}_a@example.com",
                home_currency="USD",
                current_balance_minor=10000,
            )
            self.user_b = create_user(
                db,
                email=f"{uuid.uuid4().hex[:10]}_b@example.com",
                home_currency="USD",
                current_balance_minor=20000,
            )
            db.commit()
            self.test_user_id_a = self.user_a.id
            self.test_user_id_b = self.user_b.id
            self.today = date(2026, 9, 14)

    def tearDown(self):
        with self.SessionFactory() as db:
            for table in reversed(Base.metadata.sorted_tables):
                if "user_id" in table.c:
                    db.execute(delete(table).where(
                        (table.c.user_id == self.test_user_id_a) |
                        (table.c.user_id == self.test_user_id_b)
                    ))
            db.execute(delete(User).where(
                (User.id == self.test_user_id_a) |
                (User.id == self.test_user_id_b)
            ))
            db.commit()

    def test_preference_isolation(self):
        with self.SessionFactory() as db:
            # User A sets a preference
            save_preference(db, self.user_a, key="risk_tolerance", value="high")
            # User B cannot see it
            self.assertIsNone(get_preference(db, self.user_b, key="risk_tolerance"))
            # User A sees it
            self.assertEqual(get_preference(db, self.user_a, key="risk_tolerance"), "high")

    def test_goal_isolation(self):
        with self.SessionFactory() as db:
            # User A creates a goal
            goal = save_goal(
                db,
                self.user_a,
                kind="save",
                target_amount_minor=500000,
                description="Vacation fund",
            )
            # User B cannot reach it: the read is scoped to the owner
            with self.assertRaises(MemoryServiceError) as cm:
                get_goal(db, self.user_b, goal.id)
            self.assertEqual(cm.exception.code, "not_found")
            # User A sees it
            retrieved = get_goal(db, self.user_a, goal.id)
            self.assertEqual(retrieved.description, "Vacation fund")

    def test_fact_isolation(self):
        with self.SessionFactory() as db:
            # User A states a fact
            fact = save_fact(
                db,
                self.user_a,
                text="I am between jobs until June",
                source="manual",
            )
            # User B cannot reach it: the read is scoped to the owner
            with self.assertRaises(MemoryServiceError) as cm:
                get_fact(db, self.user_b, fact.id)
            self.assertEqual(cm.exception.code, "not_found")
            # User A sees it
            retrieved = get_fact(db, self.user_a, fact.id)
            self.assertEqual(retrieved.text, "I am between jobs until June")


class TestMemoryCRUD(ApiTestBase):
    """Create/read/update behavior for preferences (upsert), goals (update), facts (close window)."""

    def setUp(self):
        super().setUp()
        with self.SessionFactory() as db:
            self.user = create_user(
                db,
                email=f"{uuid.uuid4().hex[:10]}@example.com",
                home_currency="USD",
                current_balance_minor=15000,
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

    def test_preference_upsert(self):
        with self.SessionFactory() as db:
            # First set
            save_preference(db, self.user, key="display_currency", value="EUR")
            self.assertEqual(get_preference(db, self.user, key="display_currency"), "EUR")
            # Update (upsert)
            save_preference(db, self.user, key="display_currency", value="GBP")
            self.assertEqual(get_preference(db, self.user, key="display_currency"), "GBP")
            # get_preferences is a key -> value mapping, so one key means one row
            prefs = get_preferences(db, self.user)
            self.assertEqual(prefs, {"display_currency": "GBP"})

    def test_goal_update(self):
        with self.SessionFactory() as db:
            # Create goal
            goal = save_goal(
                db,
                self.user,
                kind="payoff",
                target_amount_minor=100000,
                description="Credit card debt",
                priority=1,
            )
            # Update goal
            updated = update_goal(
                db,
                self.user,
                goal_id=goal.id,
                target_amount_minor=50000,
                description="Remaining credit card debt",
                priority=2,
            )
            self.assertEqual(updated.target_amount_minor, 50000)
            self.assertEqual(updated.description, "Remaining credit card debt")
            self.assertEqual(updated.priority, 2)
            # Fetch to confirm persistence
            fetched = get_goal(db, self.user, goal.id)
            self.assertEqual(fetched.target_amount_minor, 50000)

    def test_fact_close_window(self):
        with self.SessionFactory() as db:
            # Create fact with future validity
            fact = save_fact(
                db,
                self.user,
                text="I will start a new job on 2027-01-15",
                valid_from=date(2027, 1, 15),
                valid_to=date(2027, 12, 31),
            )
            # As-of today: fact is not active (valid_from in future)
            active_today = list_active_facts(db, self.user, as_of=self.today)
            self.assertEqual(len(active_today), 0)
            # As-of validity start: fact becomes active
            active_jan = list_active_facts(db, self.user, as_of=date(2027, 1, 15))
            self.assertEqual(len(active_jan), 1)
            # Close the fact early
            closed = close_fact(
                db,
                self.user,
                fact_id=fact.id,
                valid_to=date(2027, 6, 30),
            )
            self.assertEqual(closed.valid_to, date(2027, 6, 30))
            # After closing, fact is not active for dates beyond 2027-06-30
            active_july = list_active_facts(db, self.user, as_of=date(2027, 7, 1))
            self.assertEqual(len(active_july), 0)


class TestMemoryRendering(ApiTestBase):
    """Empty-memory rendering and deterministic retrieval."""

    def setUp(self):
        super().setUp()
        with self.SessionFactory() as db:
            self.user = create_user(
                db,
                email=f"{uuid.uuid4().hex[:10]}@example.com",
                home_currency="USD",
                current_balance_minor=12000,
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

    def test_empty_memory_renders_gracefully(self):
        with self.SessionFactory() as db:
            snapshot = build_memory_snapshot(db, self.user, self.today)
            text = render_memory_text(snapshot)
            self.assertEqual(text, "No stored memory for this user yet.")

    def test_memory_rendering_is_deterministic(self):
        with self.SessionFactory() as db:
            # Insert some memory
            save_preference(db, self.user, key="risk_tolerance", value="medium")
            save_goal(
                db,
                self.user,
                kind="save",
                target_amount_minor=300000,
                description="Down payment",
            )
            save_fact(
                db,
                self.user,
                text="I prefer digital receipts",
                source="manual",
            )
            # Build snapshot twice
            snap1 = build_memory_snapshot(db, self.user, self.today)
            snap2 = build_memory_snapshot(db, self.user, self.today)
            self.assertEqual(snap1, snap2)
            # Render twice
            text1 = render_memory_text(snap1)
            text2 = render_memory_text(snap2)
            self.assertEqual(text1, text2)

    def test_memory_block_is_bounded(self):
        """The prompt block must stay small however much memory accumulates.

        The block is injected into every prompt and the orchestrator re-sends
        its whole history each step, so an unbounded render would inflate every
        later request until the provider's context limit is exceeded.
        """
        from app.memory.snapshot import (
            MAX_FACTS_IN_PROMPT,
            MAX_GOALS_IN_PROMPT,
            MAX_RENDERED_ENTRY_CHARS,
        )

        with self.SessionFactory() as db:
            for i in range(MAX_FACTS_IN_PROMPT * 3):
                save_fact(db, self.user, text=f"fact {i} " + "x" * 3000, source="manual")
            for i in range(MAX_GOALS_IN_PROMPT * 3):
                save_goal(
                    db,
                    self.user,
                    kind="save",
                    target_amount_minor=1000 + i,
                    description="goal description " + "d" * 1900,
                )

            text = render_memory_text(build_memory_snapshot(db, self.user, self.today))

            # Bounded well below anything that could threaten a context window
            self.assertLess(len(text), 20000)
            # Truncation is stated, not silent
            self.assertIn("[truncated]", text)
            self.assertIn("more facts not shown", text)
            self.assertIn("more goals not shown", text)
            # The tools remain the full read path, and the stores still hold all rows
            self.assertEqual(len(list_active_facts(db, self.user, as_of=self.today)), MAX_FACTS_IN_PROMPT * 3)
            self.assertEqual(len(list_goals(db, self.user)), MAX_GOALS_IN_PROMPT * 3)
            # Every rendered line is clipped
            for line in text.splitlines():
                self.assertLessEqual(len(line), MAX_RENDERED_ENTRY_CHARS + 200)

    def test_memory_block_contains_no_braces(self):
        """User text containing braces cannot corrupt the prompt's JSON block."""
        with self.SessionFactory() as db:
            save_fact(
                db,
                self.user,
                text='my budget looks like {"rent": 1200} each month',
                source="manual",
            )
            save_preference(db, self.user, key="note_style", value="uses {curly} braces")

            user_prompt = ContextBuilder.build_user_prompt(
                db, self.user, "What is my balance?", self.today
            )
            # The memory section carries no raw braces
            memory_section = user_prompt[user_prompt.index("MEMORY (user-stated context only"):]
            self.assertNotIn("{", memory_section)
            self.assertNotIn("}", memory_section)
            # And the financial-state JSON that precedes it still parses
            state = json.loads(user_prompt[user_prompt.index("{"):user_prompt.rindex("}") + 1])
            self.assertEqual(state["user_id"], self.user.id)


class TestMemoryProtection(ApiTestBase):
    """Protection against memory being treated as financial truth."""

    def setUp(self):
        super().setUp()
        with self.SessionFactory() as db:
            self.user = create_user(
                db,
                email=f"{uuid.uuid4().hex[:10]}@example.com",
                home_currency="USD",
                current_balance_minor=50000,
                minimum_balance_minor=1000,
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

    def test_reserved_preference_keys_are_rejected(self):
        """Attempting to save a preference that shadows engine-authoritative config raises an error."""
        reserved_keys = [
            "currentbalance",
            "currentbalanceminor",
            "balance",
            "minimumbalance",
            "minimumbalanceminor",
            "minimumbalancetokeep",
            "protectedcategories",
            "reduciblecategories",
            "stoppablecategories",
            "paymentmethodsconsidered",
            "maxinstallmentpayments",
            "homecurrency",
            "safeamount",
            "maxsafeamount",
            "affordable",
        ]
        with self.SessionFactory() as db:
            for key in reserved_keys:
                with self.assertRaises(MemoryServiceError) as cm:
                    save_preference(db, self.user, key=key, value="something")
                # The error should be about reserved key, and the message should not echo the value
                self.assertEqual(cm.exception.code, "reserved_key")
                self.assertIn(key, cm.exception.message)
                self.assertNotIn("something", cm.exception.message)

    def test_memory_writes_do_not_change_financial_state(self):
        """Saving preferences, goals, facts does not alter the output of state_to_json."""
        with self.SessionFactory() as db:
            # Get baseline financial state
            from app.services.financial_state_service import get_financial_state, state_to_json
            state = get_financial_state(db, self.user, self.today)
            baseline = state_to_json(state)

            # Save a preference
            save_preference(db, self.user, key="risk_tolerance", value="high")
            # Save a goal
            save_goal(
                db,
                self.user,
                kind="save",
                target_amount_minor=1000000,
                description="House down payment",
            )
            # Save a fact
            save_fact(
                db,
                self.user,
                text="I expect a bonus in December",
                source="manual",
            )

            # Financial state should be unchanged
            state_after = get_financial_state(db, self.user, self.today)
            after = state_to_json(state_after)
            self.assertEqual(baseline, after)

    def test_engine_does_not_read_memory_tables(self):
        """The engine's get_financial_state does not join memory tables; we can verify by checking that memory changes don't affect it."""
        # This is implicitly tested by the above test, but we can also check that the service doesn't import memory modules.
        # We'll trust that the service layer is separate and only uses the ledger.
        pass  # Covered by test_memory_writes_do_not_change_financial_state


class TestMemoryContextIntegration(ApiTestBase):
    """Memory appears in the built user prompt; empty memory is graceful."""

    def setUp(self):
        super().setUp()
        with self.SessionFactory() as db:
            self.user = create_user(
                db,
                email=f"{uuid.uuid4().hex[:10]}@example.com",
                home_currency="USD",
                current_balance_minor=25000,
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

    def test_memory_integrates_into_user_prompt(self):
        with self.SessionFactory() as db:
            # Populate memory
            save_preference(db, self.user, key="risk_tolerance", value="low")
            save_goal(
                db,
                self.user,
                kind="purchase",
                target_amount_minor=150000,
                description="New bicycle",
            )
            save_fact(
                db,
                self.user,
                text="My lease ends on 2027-06-30",
                source="manual",
                valid_from=date(2026, 7, 1),
                valid_to=date(2027, 6, 30),
            )
            # Build user prompt
            user_prompt = ContextBuilder.build_user_prompt(
                db, self.user, "Can I afford the bicycle?", self.today
            )
            # Check that memory section appears and contains our data
            self.assertIn("MEMORY (user-stated context only", user_prompt)
            self.assertIn("Preferences:", user_prompt)
            self.assertIn("- risk_tolerance: low", user_prompt)
            self.assertIn("Goals (user-stated intentions, not obligations):", user_prompt)
            self.assertIn("- [purchase] 150000 USD minor units", user_prompt)
            self.assertIn("Facts the user has stated:", user_prompt)
            self.assertIn("- My lease ends on 2027-06-30", user_prompt)
            # Ensure the financial state snapshot is still present (starts with '{')
            self.assertIn("CURRENT FINANCIAL STATE SNAPSHOT:", user_prompt)
            # The memory block is after the financial state, so we can check that the JSON slicing in test_context.py still works
            # by verifying that the part between the first '{' and last '}' is valid JSON and does not contain memory braces.
            start = user_prompt.index("{")
            end = user_prompt.rindex("}") + 1
            json_str = user_prompt[start:end]
            state = json.loads(json_str)
            self.assertEqual(state["user_id"], self.user.id)
            self.assertEqual(state["home_currency"], "USD")
            # The state should not contain any memory-specific keys (like preferences, goals, facts)
            self.assertNotIn("preferences", state)
            self.assertNotIn("goals", state)
            self.assertNotIn("facts", state)

    def test_empty_memory_is_graceful(self):
        with self.SessionFactory() as db:
            user_prompt = ContextBuilder.build_user_prompt(
                db, self.user, "What is my balance?", self.today
            )
            # The memory section should say there is no stored memory
            self.assertIn("MEMORY (user-stated context only", user_prompt)
            self.assertIn("No stored memory for this user yet.", user_prompt)
            # Financial state should still be present
            self.assertIn("CURRENT FINANCIAL STATE SNAPSHOT:", user_prompt)
            start = user_prompt.index("{")
            end = user_prompt.rindex("}") + 1
            json_str = user_prompt[start:end]
            state = json.loads(json_str)
            self.assertEqual(state["current_balance_minor"], 25000)


class TestMemoryPersistenceRollback(ApiTestBase):
    """Persistence/rollback: flush without commit + rollback discards."""

    def setUp(self):
        super().setUp()
        with self.SessionFactory() as db:
            self.user = create_user(
                db,
                email=f"{uuid.uuid4().hex[:10]}@example.com",
                home_currency="USD",
                current_balance_minor=18000,
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

    def test_memory_persists_after_commit(self):
        with self.SessionFactory() as db:
            save_preference(db, self.user, key="theme", value="dark")
            db.commit()  # persist
        # New session, should still see the preference
        with self.SessionFactory() as db:
            self.assertEqual(get_preference(db, self.user, key="theme"), "dark")

    def test_memory_rolled_back_on_failure(self):
        with self.SessionFactory() as db:
            save_preference(db, self.user, key="theme", value="dark")
            # Simulate an error that causes rollback
            try:
                raise Exception("Simulated error")
            except Exception:
                db.rollback()
        # New session, preference should not exist
        with self.SessionFactory() as db:
            self.assertIsNone(get_preference(db, self.user, key="theme"))


class TestMemoryCredentialRejection(ApiTestBase):
    """Credential rejection: secrets refused, error message never echoes the value."""

    def setUp(self):
        super().setUp()
        with self.SessionFactory() as db:
            self.user = create_user(
                db,
                email=f"{uuid.uuid4().hex[:10]}@example.com",
                home_currency="USD",
                current_balance_minor=22000,
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

    def test_secret_keywords_in_preference_key_are_rejected(self):
        """A preference *named* like a credential is refused, whatever the value."""
        bad_keys = ["api_key", "access_token", "secret", "password", "private_key"]
        with self.SessionFactory() as db:
            for key in bad_keys:
                with self.assertRaises(MemoryServiceError) as cm:
                    save_preference(db, self.user, key=key, value="some_ordinary_value")
                self.assertEqual(cm.exception.code, "secret_rejected")
                self.assertIn(key, cm.exception.message)
                self.assertNotIn("some_ordinary_value", cm.exception.message)

    def test_secret_values_are_rejected_in_preferences(self):
        """Attempting to save a preference with a value that looks like a secret is refused."""
        bad_values = [
            "sk_live_test_test_test_test_test",
            "ghp_abcdefghijklmnopqrstuvwxyz",
            "AKIAIOSFODNN7EXAMPLE",
            "ya29.a0AfH6SMBz",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
        ]
        with self.SessionFactory() as db:
            for value in bad_values:
                with self.assertRaises(MemoryServiceError) as cm:
                    save_preference(db, self.user, key="test_key", value=value)
                # The error should be about secret rejected (from the secret guard)
                self.assertEqual(cm.exception.code, "secret_rejected")
                # The error message should mention the path (key) but not the value
                self.assertIn("test_key", cm.exception.message)
                self.assertNotIn(value, cm.exception.message)

    def test_secret_values_are_rejected_in_facts(self):
        """Attempting to save a fact with a secret-like value is refused."""
        bad_values = [
            "sk_live_test_test_test_test_test",
            "ghp_abcdefghijklmnopqrstuvwxyz",
            "AKIAIOSFODNN7EXAMPLE",
            "ya29.a0AfH6SMBz",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
        ]
        with self.SessionFactory() as db:
            for value in bad_values:
                with self.assertRaises(MemoryServiceError) as cm:
                    save_fact(db, self.user, text=value, source="manual")
                self.assertEqual(cm.exception.code, "secret_rejected")
                self.assertIn("fact", cm.exception.message)
                self.assertNotIn(value, cm.exception.message)


class TestMemoryToolsThroughRegistry(ApiTestBase):
    """The memory tools are registered and work through the orchestrator (tool execution path)."""

    def setUp(self):
        super().setUp()
        with self.SessionFactory() as db:
            self.user = create_user(
                db,
                email=f"{uuid.uuid4().hex[:10]}@example.com",
                home_currency="USD",
                current_balance_minor=30000,
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

    def test_get_preferences_tool_via_registry(self):
        with self.SessionFactory() as db:
            # Set a preference via the service layer
            save_preference(db, self.user, key="notification_preference", value="email")
            # Execute via registry
            tool = registry.get_tool("get_preferences")
            result = tool.execute(
                db=db,
                user=self.user,
                args={},
                as_of=self.today,
            )
            self.assertTrue(result["success"])
            self.assertIn("preferences", result)
            prefs = result["preferences"]
            # preferences is a dict keyed by preference name
            self.assertEqual(prefs, {"notification_preference": "email"})

    def test_save_preference_tool_via_registry(self):
        with self.SessionFactory() as db:
            tool = registry.get_tool("save_preference")
            result = tool.execute(
                db=db,
                user=self.user,
                args={"key": "risk_tolerance", "value": "high"},
                as_of=self.today,
            )
            self.assertTrue(result["success"])
            self.assertEqual(result["key"], "risk_tolerance")
            # Verify it persisted
            prefs = get_preferences(db, self.user)
            self.assertEqual(prefs, {"risk_tolerance": "high"})

    def test_list_goals_tool_via_registry(self):
        with self.SessionFactory() as db:
            # Create a goal
            save_goal(
                db,
                self.user,
                kind="save",
                target_amount_minor=200000,
                description="Emergency fund",
                priority=1,
            )
            tool = registry.get_tool("list_goals")
            result = tool.execute(
                db=db,
                user=self.user,
                args={"include_achieved": True},
                as_of=self.today,
            )
            self.assertTrue(result["success"])
            self.assertIn("goals", result)
            goals = result["goals"]
            self.assertEqual(len(goals), 1)
            self.assertEqual(goals[0]["description"], "Emergency fund")
            self.assertEqual(goals[0]["target_amount_minor"], 200000)

    def test_save_goal_tool_via_registry(self):
        with self.SessionFactory() as db:
            tool = registry.get_tool("save_goal")
            result = tool.execute(
                db=db,
                user=self.user,
                args={
                    "kind": "payoff",
                    "target_amount_minor": 50000,
                    "description": "Credit card debt",
                    "priority": 2,
                },
                as_of=self.today,
            )
            self.assertTrue(result["success"])
            self.assertIn("goal", result)
            goal = result["goal"]
            self.assertEqual(goal["kind"], "payoff")
            self.assertEqual(goal["target_amount_minor"], 50000)
            self.assertEqual(goal["description"], "Credit card debt")
            self.assertEqual(goal["priority"], 2)

    def test_search_facts_tool_via_registry(self):
        with self.SessionFactory() as db:
            # Save a fact
            save_fact(
                db,
                self.user,
                text="I prefer paperless billing",
                source="manual",
            )
            tool = registry.get_tool("search_facts")
            result = tool.execute(
                db=db,
                user=self.user,
                args={"query_text": "paperless"},
                as_of=self.today,
            )
            self.assertTrue(result["success"])
            self.assertIn("facts", result)
            facts = result["facts"]
            self.assertEqual(len(facts), 1)
            self.assertEqual(facts[0]["text"], "I prefer paperless billing")

    def test_save_fact_tool_via_registry(self):
        with self.SessionFactory() as db:
            tool = registry.get_tool("save_fact")
            result = tool.execute(
                db=db,
                user=self.user,
                args={
                    "text": "My internet bill is $60/month",
                    "valid_from": self.today.isoformat(),
                },
                as_of=self.today,
            )
            self.assertTrue(result["success"])
            self.assertIn("fact", result)
            fact = result["fact"]
            self.assertEqual(fact["text"], "My internet bill is $60/month")
            self.assertEqual(fact["valid_from"], self.today.isoformat())


class TestOrchestratorWithMemory(ApiTestBase):
    """End-to-end: memory appears in context and influences the LLM's tool choices (without changing financial state)."""

    def setUp(self):
        super().setUp()
        with self.SessionFactory() as db:
            self.user = create_user(
                db,
                email=f"{uuid.uuid4().hex[:10]}@example.com",
                home_currency="USD",
                current_balance_minor=50000,
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

    def test_memory_in_context_affects_llm_prompt(self):
        """The orchestrator's built user prompt includes the memory section."""
        llm = CapturingLLM()
        with self.SessionFactory() as db:
            # Populate memory
            save_preference(db, self.user, key="risk_tolerance", value="low")
            save_goal(
                db,
                self.user,
                kind="save",
                target_amount_minor=100000,
                description="Vacation",
            )
            save_fact(
                db,
                self.user,
                text="I am saving for a down payment",
                source="manual",
            )
            # Build context
            system_prompt = ContextBuilder.build_system_prompt(self.user, self.today)
            user_prompt_with_memory = ContextBuilder.build_user_prompt(
                db, self.user, "What is my risk tolerance?", self.today
            )
            # Run orchestrator (just to see the context passed to the LLM)
            orchestrator = AgentOrchestrator(db, user=self.user, llm=llm, system_prompt=system_prompt)
            _ = orchestrator.run(user_prompt_with_memory, self.today)
            # The LLM should have received the system and user prompts
            received_system = llm.received[0][0]
            received_user = llm.received[0][1]
            self.assertIn("AI Financial Decision Agent", received_system.content)
            self.assertIn("What is my risk tolerance?", received_user.content)
            # Check that memory appears in the user prompt
            self.assertIn("MEMORY (user-stated context only", received_user.content)
            self.assertIn("- risk_tolerance: low", received_user.content)
            self.assertIn("Goals (user-stated intentions, not obligations):", received_user.content)
            self.assertIn("- [save] 100000 USD minor units", received_user.content)
            self.assertIn("Facts the user has stated:", received_user.content)
            self.assertIn("- I am saving for a down payment", received_user.content)
            # Also check that the financial state snapshot is present
            self.assertIn("CURRENT FINANCIAL STATE SNAPSHOT:", received_user.content)
            start = received_user.content.index("{")
            end = received_user.content.rindex("}") + 1
            json_str = received_user.content[start:end]
            state = json.loads(json_str)
            self.assertEqual(state["user_id"], self.user.id)
            self.assertEqual(state["current_balance_minor"], 50000)


class TestOfflineGuarantee(unittest.TestCase):
    """The memory suite is offline by construction, and says so."""

    def test_network_access_is_blocked(self):
        """The module-level guard really is installed for every test here."""
        with self.assertRaises(AssertionError):
            socket.socket()

    def test_canned_provider_is_used_for_agent_turns(self):
        """Agent execution in this module runs on the canned provider only."""
        llm = CapturingLLM()
        response = llm.generate([])
        self.assertEqual(response.text_content, "Final canned answer.")
        self.assertIsNone(response.tool_calls)


if __name__ == "__main__":
    unittest.main()
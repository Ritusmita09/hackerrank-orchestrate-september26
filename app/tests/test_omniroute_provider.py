"""Mocked unit tests for the OmniRoute provider adapter (Phase 4).

Every test runs against an ``httpx.MockTransport`` — no socket is opened, no
real LLM is called, and no credential ever leaves the test process. The mock
replies use the exact wire shapes observed from the live OmniRoute instance
(an Anthropic Messages API), including its ``thinking`` blocks.
"""
import json
import os
import unittest
from datetime import date
from typing import Any, Dict, List
from unittest import mock

import httpx
import uuid
from sqlalchemy import delete

from app.config import get_settings
from app.db.models import Base, User
from app.services.user_service import create_user
from app.tests.test_api import ApiTestBase
from app.agent.context import ContextBuilder
from app.agent.orchestrator import (
    AgentOrchestrator,
    LLMResponse,
    Message,
    Role,
    ToolCall,
)
from app.agent.llm.omniroute_provider import OmniRouteError, OmniRouteProvider
from app.agent.llm.factory import create_llm_provider

# An obviously-fake credential: never a real secret, but long/structured
# enough that leakage into an error message would be visible in a test.
FAKE_API_KEY = "test-key-definitely-not-real-1234567890"
BASE_URL = "http://omniroute.test"
MODEL = "agentrouter/deepseek-v4-flash"


def make_provider(handler, **kwargs) -> OmniRouteProvider:
    """Build a provider whose HTTP traffic is answered by ``handler``."""
    defaults = dict(
        base_url=BASE_URL,
        api_key=FAKE_API_KEY,
        model=MODEL,
        tools=[{
            "name": "get_financial_state",
            "description": "Get the current financial state",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }],
    )
    defaults.update(kwargs)
    return OmniRouteProvider(transport=httpx.MockTransport(handler), **defaults)


def text_response(*texts: str) -> Dict[str, Any]:
    """A well-formed Anthropic response containing only text blocks."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": t} for t in texts],
    }


def tool_use_response(*calls: Dict[str, Any]) -> Dict[str, Any]:
    """A well-formed Anthropic response containing only tool_use blocks."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["input"]}
            for c in calls
        ],
    }


def capture_handler(response_body: Any, status_code: int = 200, requests: list = None):
    """Handler that records the outgoing request and returns a canned body."""
    if requests is None:
        requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status_code, json=response_body)

    return handler


class TestOmniRouteProviderBasics(unittest.TestCase):

    def test_missing_configuration(self):
        """No base URL or model means a clear, credential-free error."""
        env = {"OMNIROUTE_BASE_URL": "", "OMNIROUTE_API_KEY": "", "OMNIROUTE_MODEL": ""}
        with mock.patch.dict("os.environ", env, clear=False):
            with self.assertRaises(OmniRouteError) as ctx:
                OmniRouteProvider(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
            self.assertIn("OMNIROUTE_BASE_URL", str(ctx.exception))

            with self.assertRaises(OmniRouteError) as ctx:
                OmniRouteProvider(
                    base_url=BASE_URL,
                    transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
                )
            self.assertIn("OMNIROUTE_MODEL", str(ctx.exception))

    def test_configuration_from_environment(self):
        """Constructor falls back to the documented environment variables."""
        env = {
            "OMNIROUTE_BASE_URL": BASE_URL + "/",
            "OMNIROUTE_API_KEY": FAKE_API_KEY,
            "OMNIROUTE_MODEL": MODEL,
        }
        with mock.patch.dict("os.environ", env, clear=False):
            provider = OmniRouteProvider(
                transport=httpx.MockTransport(lambda r: httpx.Response(200, json=text_response("ok")))
            )
            self.assertEqual(provider.base_url, BASE_URL)  # trailing slash stripped
            self.assertEqual(provider.model, MODEL)
            provider.close()


class TestRequestTranslation(unittest.TestCase):
    """The internal Message list must become a valid Anthropic request."""

    def _captured_payload(self, messages: List[Message]) -> Dict[str, Any]:
        requests: List[httpx.Request] = []
        provider = make_provider(capture_handler(text_response("ok"), requests=requests))
        try:
            provider.generate(messages)
        finally:
            provider.close()
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request.url.host, "omniroute.test")
        self.assertEqual(request.url.path, "/v1/messages")
        self.assertEqual(request.headers["x-api-key"], FAKE_API_KEY)
        self.assertEqual(request.headers["anthropic-version"], "2023-06-01")
        return json.loads(request.content)

    def test_system_user_assistant_tool_translation(self):
        """Full round-trip: every internal role lands in the right wire slot."""
        messages = [
            Message(role=Role.SYSTEM, content="You are a financial agent."),
            Message(role=Role.USER, content="What is my balance?"),
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[ToolCall(id="call_1", name="get_financial_state", arguments={})],
            ),
            Message(role=Role.TOOL, tool_call_id="call_1", content='{"balance": 100}'),
        ]
        payload = self._captured_payload(messages)

        # SYSTEM becomes the top-level system string, not a message role.
        self.assertEqual(payload["system"], "You are a financial agent.")
        self.assertEqual(payload["model"], MODEL)
        self.assertEqual(
            payload["messages"],
            [
                {"role": "user", "content": [{"type": "text", "text": "What is my balance?"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "call_1", "name": "get_financial_state", "input": {}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": '{"balance": 100}',
                        }
                    ],
                },
            ],
        )

    def test_tool_schemas_are_transmitted(self):
        """The configured tool schemas must be advertised in the request."""
        payload = self._captured_payload([Message(role=Role.USER, content="hi")])
        self.assertEqual(
            payload["tools"],
            [{
                "name": "get_financial_state",
                "description": "Get the current financial state",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            }],
        )

    def test_error_tool_results_flagged(self):
        """An error tool result must carry is_error so the model can recover."""
        payload = self._captured_payload([
            Message(role=Role.ASSISTANT, tool_calls=[
                ToolCall(id="c1", name="hack", arguments={})
            ]),
            Message(role=Role.TOOL, tool_call_id="c1", content="Error: KeyError", is_error=True),
        ])
        tool_result = payload["messages"][-1]["content"][0]
        self.assertTrue(tool_result["is_error"])

    def test_parallel_tool_results_merge_into_one_user_turn(self):
        """Multiple tool results become a single user message of blocks."""
        payload = self._captured_payload([
            Message(role=Role.ASSISTANT, tool_calls=[
                ToolCall(id="c1", name="a", arguments={}),
                ToolCall(id="c2", name="b", arguments={}),
            ]),
            Message(role=Role.TOOL, tool_call_id="c1", content="1"),
            Message(role=Role.TOOL, tool_call_id="c2", content="2"),
        ])
        self.assertEqual(len(payload["messages"]), 2)
        merged = payload["messages"][-1]
        self.assertEqual(merged["role"], "user")
        self.assertEqual([b["tool_use_id"] for b in merged["content"]], ["c1", "c2"])


class TestResponseParsing(unittest.TestCase):

    def test_normal_text_response(self):
        provider = make_provider(capture_handler(text_response("Your balance is $10.")))
        try:
            response = provider.generate([Message(role=Role.USER, content="?")])
        finally:
            provider.close()
        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.text_content, "Your balance is $10.")
        self.assertIsNone(response.tool_calls)

    def test_multiple_text_blocks_join(self):
        provider = make_provider(capture_handler(text_response("Part one. ", "Part two.")))
        try:
            response = provider.generate([Message(role=Role.USER, content="?")])
        finally:
            provider.close()
        self.assertEqual(response.text_content, "Part one. \nPart two.")

    def test_tool_call_parsing(self):
        body = tool_use_response({"id": "call_00_x", "name": "get_financial_state", "input": {}})
        provider = make_provider(capture_handler(body))
        try:
            response = provider.generate([Message(role=Role.USER, content="?")])
        finally:
            provider.close()
        self.assertIsNone(response.text_content)
        self.assertEqual(len(response.tool_calls), 1)
        call = response.tool_calls[0]
        self.assertEqual(call.id, "call_00_x")
        self.assertEqual(call.name, "get_financial_state")
        self.assertEqual(call.arguments, {})

    def test_multiple_tool_calls(self):
        body = tool_use_response(
            {"id": "c1", "name": "get_financial_state", "input": {}},
            {"id": "c2", "name": "run_forecast", "input": {"horizon_days": 30}},
        )
        provider = make_provider(capture_handler(body))
        try:
            response = provider.generate([Message(role=Role.USER, content="?")])
        finally:
            provider.close()
        self.assertEqual(
            [(c.id, c.name) for c in response.tool_calls],
            [("c1", "get_financial_state"), ("c2", "run_forecast")],
        )
        self.assertEqual(response.tool_calls[1].arguments, {"horizon_days": 30})

    def test_thinking_blocks_ignored_mixed_with_text_and_tools(self):
        """The live router emits 'thinking' blocks alongside useful output."""
        body = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "The user wants their balance..."},
                {"type": "text", "text": "I'll check your current financial state."},
                {"type": "tool_use", "id": "call_00_b", "name": "get_financial_state", "input": {}},
            ],
        }
        provider = make_provider(capture_handler(body))
        try:
            response = provider.generate([Message(role=Role.USER, content="?")])
        finally:
            provider.close()
        self.assertEqual(response.text_content, "I'll check your current financial state.")
        self.assertEqual(response.tool_calls[0].name, "get_financial_state")

    def test_plain_string_content(self):
        """Some gateways collapse content to a bare string; accept it."""
        provider = make_provider(capture_handler({"content": "Plain reply."}))
        try:
            response = provider.generate([Message(role=Role.USER, content="?")])
        finally:
            provider.close()
        self.assertEqual(response.text_content, "Plain reply.")

    def test_malformed_response_variants(self):
        """Structurally broken payloads become OmniRouteError, never a crash."""
        # Structurally broken payloads become OmniRouteError, never a crash.
        error_cases = [
            {},                                   # no content key
            {"content": None},                    # null content
            {"content": 42},                      # wrong type
            {"error": {"message": "boom"}},       # error payload on HTTP 200
        ]
        for body in error_cases:
            provider = make_provider(capture_handler(body))
            with self.subTest(body=body):
                try:
                    with self.assertRaises(OmniRouteError):
                        provider.generate([Message(role=Role.USER, content="?")])
                finally:
                    provider.close()

        # A text block with no usable text yields no content, not an error.
        provider = make_provider(capture_handler({"content": [{"type": "text"}]}))
        try:
            response = provider.generate([Message(role=Role.USER, content="?")])
            self.assertIsNone(response.text_content)
            self.assertIsNone(response.tool_calls)
        finally:
            provider.close()

        # A tool_use block with no name is undispatchable and must be dropped.
        provider = make_provider(capture_handler({"content": [{"type": "tool_use", "id": "x", "input": {}}]}))
        try:
            response = provider.generate([Message(role=Role.USER, content="?")])
            self.assertIsNone(response.tool_calls)
        finally:
            provider.close()

    def test_non_json_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>gateway error page</html>")

        provider = make_provider(handler)
        try:
            with self.assertRaises(OmniRouteError) as ctx:
                provider.generate([Message(role=Role.USER, content="?")])
            self.assertIn("non-JSON", str(ctx.exception))
        finally:
            provider.close()

    def test_provider_http_error(self):
        body = {"error": {"type": "invalid_api_key", "message": "Invalid API key"}}
        provider = make_provider(capture_handler(body, status_code=401))
        try:
            with self.assertRaises(OmniRouteError) as ctx:
                provider.generate([Message(role=Role.USER, content="?")])
            self.assertIn("HTTP 401", str(ctx.exception))
            self.assertIn("Invalid API key", str(ctx.exception))
        finally:
            provider.close()

    def test_connection_failure(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused", request=request)

        provider = make_provider(handler)
        try:
            with self.assertRaises(OmniRouteError) as ctx:
                provider.generate([Message(role=Role.USER, content="?")])
            self.assertIn("Could not reach OmniRoute", str(ctx.exception))
        finally:
            provider.close()


class TestSecretSafety(unittest.TestCase):

    def test_api_key_never_appears_in_errors(self):
        """A credential echoed back by the provider must be scrubbed."""
        leaky = {
            "error": {"type": "auth_error", "message": f"key {FAKE_API_KEY} rejected"}
        }
        provider = make_provider(capture_handler(leaky, status_code=403))
        try:
            with self.assertRaises(OmniRouteError) as ctx:
                provider.generate([Message(role=Role.USER, content="?")])
            self.assertNotIn(FAKE_API_KEY, str(ctx.exception))
            self.assertIn("***redacted***", str(ctx.exception))
        finally:
            provider.close()

    def test_api_key_not_in_connection_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError(f"proxy auth {FAKE_API_KEY} failed", request=request)

        provider = make_provider(handler)
        try:
            with self.assertRaises(OmniRouteError) as ctx:
                provider.generate([Message(role=Role.USER, content="?")])
            self.assertNotIn(FAKE_API_KEY, str(ctx.exception))
        finally:
            provider.close()

    def test_repr_and_attributes_do_not_leak_beyond_the_holder(self):
        """The key is held privately; the provider's text form stays clean."""
        provider = make_provider(capture_handler(text_response("ok")))
        try:
            self.assertNotIn(FAKE_API_KEY, repr(provider))
            self.assertTrue(provider._api_key)  # held, but not exposed in repr
        finally:
            provider.close()


class TestFactorySelection(unittest.TestCase):

    def test_none_provider_returns_none(self):
        settings = mock.Mock(llm_provider="none")
        self.assertIsNone(create_llm_provider(settings))

    def test_unknown_provider_raises(self):
        settings = mock.Mock(llm_provider="carrier-pigeon")
        with self.assertRaises(ValueError):
            create_llm_provider(settings)

    def test_omniroute_provider_built_from_settings(self):
        settings = mock.Mock(
            llm_provider="omniroute",
            omniroute_base_url=BASE_URL,
            omniroute_api_key=FAKE_API_KEY,
            omniroute_model=MODEL,
            llm_max_tokens_per_turn=1234,
        )
        provider = create_llm_provider(settings, tools=[])
        try:
            self.assertIsInstance(provider, OmniRouteProvider)
            self.assertEqual(provider.model, MODEL)
            self.assertEqual(provider.max_tokens, 1234)
        finally:
            provider.close()

    def test_omniroute_missing_model_raises(self):
        settings = mock.Mock(
            llm_provider="omniroute",
            omniroute_base_url=BASE_URL,
            omniroute_api_key=FAKE_API_KEY,
            omniroute_model="",
            llm_max_tokens_per_turn=1234,
        )
        with self.assertRaises(OmniRouteError) as ctx:
            create_llm_provider(settings, tools=[])
        self.assertIn("OMNIROUTE_MODEL", str(ctx.exception))


def capture_handler_q(bodies: List[Any], requests: List[httpx.Request]):
    """Handler that replays one body per request, in order."""
    state = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = bodies[state["i"]]
        state["i"] += 1
        return httpx.Response(200, json=body)

    return handler


class TestOrchestratorIntegrationOmniRoute(ApiTestBase):
    """The adapter must drive the unchanged orchestrator end to end.

    This is the integration contract: context assembly and the orchestrator
    are untouched, and only the provider differs from the canned-LLM tests.
    """

    def setUp(self):
        with self.SessionFactory() as db:
            user = create_user(
                db,
                email=f"{uuid.uuid4().hex[:10]}@example.com",
                home_currency="USD",
                current_balance_minor=10000,
            )
            db.commit()
            self.test_user_id = user.id
        self.today = date(2026, 9, 14)

    def tearDown(self):
        with self.SessionFactory() as db:
            for table in reversed(Base.metadata.sorted_tables):
                if "user_id" in table.c:
                    db.execute(delete(table).where(table.c.user_id == self.test_user_id))
            db.execute(delete(User).where(User.id == self.test_user_id))
            db.commit()

    def test_full_agent_turn_over_mocked_wire(self):
        """Context -> orchestrator -> adapter -> mock OmniRoute -> registry."""
        requests: List[httpx.Request] = []
        provider = OmniRouteProvider(
            base_url=BASE_URL,
            api_key=FAKE_API_KEY,
            model=MODEL,
            tools=[{
                "name": "get_financial_state",
                "description": "Get the current financial state",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            }],
            transport=httpx.MockTransport(capture_handler_q(
                [
                    # Turn 1: the model asks for the financial state.
                    tool_use_response({"id": "call_1", "name": "get_financial_state", "input": {}}),
                    # Turn 2: it answers with a final text response.
                    text_response("Your balance is $100.00."),
                ],
                requests,
            )),
        )

        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            system_prompt = ContextBuilder.build_system_prompt(user, self.today)
            user_prompt = ContextBuilder.build_user_prompt(db, user, "What is my balance?", self.today)

            orchestrator = AgentOrchestrator(
                db, user, provider, system_prompt=system_prompt, max_steps=5
            )
            try:
                final = orchestrator.run(user_prompt, self.today)
            finally:
                provider.close()

            self.assertEqual(final, "Your balance is $100.00.")

            # The wire saw two calls; the second carried the real tool result
            # produced by the Phase 2 service through the registry.
            self.assertEqual(len(requests), 2)
            second_payload = json.loads(requests[1].content)
            tool_results = [
                block
                for m in second_payload["messages"]
                if m["role"] == "user"
                for block in m["content"]
                if block.get("type") == "tool_result"
            ]
            self.assertEqual(len(tool_results), 1)
            self.assertIn('"current_balance_minor": 10000', tool_results[0]["content"])

            # Roles and registry dispatch behaved exactly as with the canned LLM.
            self.assertEqual(
                [m.role for m in orchestrator.messages],
                [Role.SYSTEM, Role.USER, Role.ASSISTANT, Role.TOOL, Role.ASSISTANT],
            )


@unittest.skipUnless(
    os.environ.get("OMNIROUTE_LIVE_TEST") == "1",
    "live OmniRoute smoke test is opt-in: set OMNIROUTE_LIVE_TEST=1 to run it",
)
class TestOmniRouteLiveSmoke(ApiTestBase):
    """Opt-in live check against a real OmniRoute instance.

    Deliberately OUTSIDE the offline suite: ``python run_tests.py`` skips it,
    so the repository's tests never call a model, never need a credential, and
    never depend on a local service being up. Run it explicitly when you want
    to verify the real wire path end to end:

        OMNIROUTE_LIVE_TEST=1 python -m unittest app.tests.test_omniroute_provider.TestOmniRouteLiveSmoke -v

    Requires OMNIROUTE_BASE_URL and OMNIROUTE_MODEL (and OMNIROUTE_API_KEY if
    the route needs one) in the environment. No secret is printed.
    """

    def test_live_turn_through_omniroute(self):
        settings = get_settings()
        provider = create_llm_provider(settings)
        if provider is None:
            self.skipTest("LLM_PROVIDER is not set to omniroute")

        with self.SessionFactory() as db:
            user = create_user(
                db,
                email=f"{uuid.uuid4().hex[:10]}@example.com",
                home_currency="USD",
                current_balance_minor=10000,
            )
            db.commit()
            try:
                orchestrator = AgentOrchestrator(
                    db,
                    user,
                    provider,
                    system_prompt=ContextBuilder.build_system_prompt(user, date(2026, 9, 14)),
                )
                answer = orchestrator.run(
                    ContextBuilder.build_user_prompt(
                        db, user, "What is my current balance?", date(2026, 9, 14)
                    ),
                    date(2026, 9, 14),
                )
                # A live model must still answer through the deterministic tools.
                self.assertTrue(answer.strip())
                self.assertEqual(orchestrator.messages[0].role, Role.SYSTEM)
            finally:
                for table in reversed(Base.metadata.sorted_tables):
                    if "user_id" in table.c:
                        db.execute(delete(table).where(table.c.user_id == user.id))
                db.execute(delete(User).where(User.id == user.id))
                db.commit()
                close = getattr(provider, "close", None)
                if callable(close):
                    close()


if __name__ == "__main__":
    unittest.main()

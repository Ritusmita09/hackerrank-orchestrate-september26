"""Extraction integration tests (Phase 4: multimodal document understanding).

Covers the parts that sit *around* the extraction service: on-disk document
storage, the OmniRoute wire translation of image attachments, extraction
provider routing, the agent tool, and the HTTP routes. Like
``test_extraction.py``, this module blocks network access at import time so an
accidental real call is impossible. The block here targets outbound connects
rather than socket construction: the FastAPI TestClient on Windows needs
``socket.socketpair`` for its event loop, which is a local pipe, not a network
call — and the provider test uses ``httpx.MockTransport``, which never connects.
"""
import base64
import io
import json
import socket
import tempfile
import threading
import unittest
import uuid
from datetime import date
from types import SimpleNamespace
from typing import Any, List
from unittest import mock

import httpx
from sqlalchemy import delete

from app.db.models import Base, User
from app.tests.test_api import ApiTestBase
from app.services.user_service import create_user

_ORIGINAL_CONNECT = socket.socket.connect
_ORIGINAL_CONNECT_EX = socket.socket.connect_ex
_ORIGINAL_SOCKETPAIR = socket.socketpair

# socket.socketpair() on Windows is emulated with a bind+listen+connect to
# 127.0.0.1 — a local pipe, not a network call. The connect inside that
# emulation is the only connect this suite ever permits.
_SOCKETPAIR_DEPTH = threading.local()


def _no_connect(self, address, *args, **kwargs):
    if getattr(_SOCKETPAIR_DEPTH, "on", False):
        return _ORIGINAL_CONNECT(self, address, *args, **kwargs)
    raise AssertionError(
        "outbound network access is not permitted in the offline extraction "
        "test suite; use a canned LLM (or the opt-in live smoke test) for "
        "real calls."
    )


def _guarded_socketpair(*args, **kwargs):
    _SOCKETPAIR_DEPTH.on = True
    try:
        return _ORIGINAL_SOCKETPAIR(*args, **kwargs)
    finally:
        _SOCKETPAIR_DEPTH.on = False


def setUpModule():
    socket.socket.connect = _no_connect
    socket.socket.connect_ex = _no_connect
    socket.socketpair = _guarded_socketpair


def tearDownModule():
    socket.socket.connect = _ORIGINAL_CONNECT
    socket.socket.connect_ex = _ORIGINAL_CONNECT_EX
    socket.socketpair = _ORIGINAL_SOCKETPAIR


def valid_draft_json() -> str:
    return json.dumps({
        "document_type": "bill",
        "issuer": "City Power",
        "issue_date": "2026-08-01",
        "due_date": "2026-08-20",
        "currency": "USD",
        "total_amount_minor": 8500,
        "line_items": [
            {"description": "Electricity", "amount_minor": 7000},
            {"description": "Service fee", "amount_minor": 1500},
        ],
        "field_confidence": {"total_amount_minor": 0.95},
        "notes": "",
    })


class CannedLLM:
    """Scripted replies with a model name; records received message lists."""

    def __init__(self, replies: List[str]):
        self.replies = list(replies)
        self.received: List[List[Any]] = []
        self.model = "test/extraction-model"

    def generate(self, messages):
        self.received.append(list(messages))
        if not self.replies:
            raise AssertionError("canned LLM ran out of scripted replies")
        class _R:
            text_content = self.replies.pop(0)
            tool_calls = None
        return _R()


class IntegrationBase(ApiTestBase):
    """ApiTestBase plus a per-test user and a temp storage root."""

    def setUp(self):
        super().setUp()
        with self.SessionFactory() as db:
            self.user = create_user(
                db,
                email=f"{uuid.uuid4().hex[:10]}@example.com",
                home_currency="USD",
                current_balance_minor=10000,
            )
            db.commit()
            self.test_user_id = self.user.id
        self.storage_root = tempfile.mkdtemp(prefix="extraction-int-")
        self.as_of = date(2026, 9, 14)

    def tearDown(self):
        with self.SessionFactory() as db:
            for table in reversed(Base.metadata.sorted_tables):
                if "user_id" in table.c:
                    db.execute(delete(table).where(table.c.user_id == self.test_user_id))
            db.execute(delete(User).where(User.id == self.test_user_id))
            db.commit()

    def fake_settings(self) -> Any:
        """Settings stub pointing every storage lookup at the temp root."""
        return SimpleNamespace(
            storage_dir=self.storage_root,
            max_upload_bytes=5_242_880,
            allowed_mime_types=[
                "text/csv", "application/pdf", "image/png",
                "image/jpeg", "text/plain",
            ],
        )


# --- document storage ---------------------------------------------------------

class TestDocumentStorage(unittest.TestCase):

    def test_round_trip_and_hash_addressing(self):
        from app.services.document_service import (
            read_document_content, write_document_content,
        )

        content = b"\x89PNG round-trip"
        digest = __import__("hashlib").sha256(content).hexdigest()
        with tempfile.TemporaryDirectory() as root:
            path = write_document_content("user-1", digest, content, storage_root=root)
            # The stored bytes hash to the name of the file they live in.
            self.assertTrue(path.endswith(digest))
            self.assertIn("user-1", path.replace("\\", "/").split("/")[-2])
            self.assertEqual(
                read_document_content("user-1", digest, storage_root=root), content
            )

    def test_same_bytes_overwrite_same_file(self):
        from app.services.document_service import write_document_content

        with tempfile.TemporaryDirectory() as root:
            p1 = write_document_content("u", "h1", b"one", storage_root=root)
            p2 = write_document_content("u", "h1", b"one", storage_root=root)
            self.assertEqual(p1, p2)

    def test_missing_content_raises(self):
        from app.services.document_service import (
            InputServiceError, read_document_content,
        )

        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(InputServiceError) as cm:
                read_document_content("u", "missing", storage_root=root)
            self.assertEqual(cm.exception.code, "document_content_missing")

    def test_users_have_separate_directories(self):
        import os

        from app.services.document_service import write_document_content

        with tempfile.TemporaryDirectory() as root:
            write_document_content("alice", "h", b"x", storage_root=root)
            write_document_content("bob", "h", b"y", storage_root=root)
            self.assertTrue(os.path.isdir(os.path.join(root, "alice")))
            self.assertTrue(os.path.isdir(os.path.join(root, "bob")))


# --- wire format: attachments become base64 image blocks ----------------------

class TestAttachmentWireFormat(unittest.TestCase):

    def _provider(self, handler) -> "Any":
        from app.agent.llm.omniroute_provider import OmniRouteProvider

        transport = httpx.MockTransport(handler)
        return OmniRouteProvider(
            base_url="http://omniroute.test",
            api_key="test-key",
            model="test/model",
            tools=[],
            transport=transport,
        )

    def test_image_blocks_precede_text(self):
        from app.agent.orchestrator import Attachment, Message, Role

        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(200, json={
                "content": [{"type": "text", "text": "ok"}]
            })

        payload_bytes = b"\x89PNG wire-bytes"
        provider = self._provider(handler)
        try:
            provider.generate([
                Message(
                    role=Role.USER,
                    content="Extract this document.",
                    attachments=[Attachment(
                        media_type="image/png",
                        data_b64=base64.b64encode(payload_bytes).decode("ascii"),
                    )],
                )
            ])
        finally:
            provider.close()

        wire = captured["payload"]["messages"]
        self.assertEqual(len(wire), 1)
        self.assertEqual(wire[0]["role"], "user")
        blocks = wire[0]["content"]
        self.assertEqual(len(blocks), 2)
        # Image block first (the API expects it before the referencing text).
        self.assertEqual(blocks[0]["type"], "image")
        self.assertEqual(blocks[0]["source"]["type"], "base64")
        self.assertEqual(blocks[0]["source"]["media_type"], "image/png")
        self.assertEqual(
            base64.b64decode(blocks[0]["source"]["data"]), payload_bytes
        )
        self.assertEqual(blocks[1], {"type": "text", "text": "Extract this document."})
        # Extraction turns advertise no tools.
        self.assertNotIn("tools", captured["payload"])


# --- provider routing ---------------------------------------------------------

class TestExtractionProviderRouting(unittest.TestCase):

    def _settings(self, **overrides: Any) -> Any:
        base = dict(
            llm_provider="omniroute",
            omniroute_base_url="http://omniroute.test",
            omniroute_api_key="test-key",
            omniroute_model="default/model",
            omniroute_extraction_model="",
            llm_max_tokens_per_turn=1234,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_none_provider_when_disabled(self):
        from app.agent.llm.factory import create_extraction_provider

        self.assertIsNone(
            create_extraction_provider(self._settings(llm_provider="none"))
        )

    def test_falls_back_to_default_model(self):
        from app.agent.llm.factory import create_extraction_provider

        provider = create_extraction_provider(self._settings())
        try:
            self.assertEqual(provider.model, "default/model")
            self.assertEqual(provider.tools, [])   # extraction: no tools advertised
            self.assertEqual(provider.max_tokens, 1234)
        finally:
            provider.close()

    def test_dedicated_extraction_model_wins(self):
        from app.agent.llm.factory import create_extraction_provider

        provider = create_extraction_provider(
            self._settings(omniroute_extraction_model="vision/model")
        )
        try:
            self.assertEqual(provider.model, "vision/model")
        finally:
            provider.close()


# --- the agent tool -----------------------------------------------------------

class TestExtractDocumentTool(IntegrationBase):

    def _run_tool(self, llm, document_id: str) -> Any:
        from app.agent.tools import registry

        # The tool passes no storage_root, so storage lookups go through
        # get_settings() — point that at this test's temp root.
        with mock.patch(
            "app.services.document_service.get_settings",
            return_value=self.fake_settings(),
        ):
            with self.SessionFactory() as db:
                user = db.get(User, self.test_user_id)
                result = registry.execute(
                    "extract_document", db, user,
                    {"document_id": document_id}, self.as_of,
                )
                db.commit()
        return result

    def _upload(self, content: bytes = b"\x89PNG tool-bytes") -> str:
        from app.services.document_service import register_document

        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            doc = register_document(
                db, user, filename="bill.png", mime_type="image/png",
                content=content, storage_root=self.storage_root,
            )
            db.commit()
            return doc.id

    def test_tool_returns_draft_and_note(self):
        doc_id = self._upload()
        llm = CannedLLM([valid_draft_json()])
        with mock.patch(
            "app.agent.llm.factory.create_extraction_provider", return_value=llm
        ):
            result = self._run_tool(llm, doc_id)

        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "draft")
        self.assertFalse(result["cached"])
        self.assertEqual(result["draft"]["total_amount_minor"], 8500)
        self.assertIn("confirm", result["note"])

    def test_tool_reports_unconfigured_llm(self):
        doc_id = self._upload()
        with mock.patch(
            "app.agent.llm.factory.create_extraction_provider", return_value=None
        ):
            result = self._run_tool(None, doc_id)

        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "llm_not_configured")

    def test_tool_maps_domain_errors(self):
        with mock.patch(
            "app.agent.llm.factory.create_extraction_provider",
            return_value=CannedLLM([]),
        ):
            result = self._run_tool(None, "no-such-doc")

        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "document_not_found")

    def test_tool_maps_transport_errors(self):
        from app.agent.llm.omniroute_provider import OmniRouteError

        doc_id = self._upload()

        class ExplodingLLM:
            model = "test/exploding"

            def generate(self, messages):
                raise OmniRouteError("connection refused (scrubbed)")

        with mock.patch(
            "app.agent.llm.factory.create_extraction_provider",
            return_value=ExplodingLLM(),
        ):
            result = self._run_tool(None, doc_id)

        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "llm_error")

    def test_tool_requires_document_id_argument(self):
        from app.agent.tools import registry

        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            with self.assertRaises(Exception):
                registry.execute("extract_document", db, user, {}, self.as_of)
            db.rollback()


# --- HTTP routes --------------------------------------------------------------

class TestExtractionRoutes(IntegrationBase):

    def _upload_png(self) -> str:
        with mock.patch(
            "app.services.document_service.get_settings",
            return_value=self.fake_settings(),
        ):
            resp = self.client.post(
                f"/users/{self.test_user_id}/documents",
                files={"file": ("bill.png", io.BytesIO(b"\x89PNG route-bytes"), "image/png")},
                params={"kind": "bill"},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()["id"]

    def _extract(self, doc_id: str, llm):
        with mock.patch(
            "app.services.document_service.get_settings",
            return_value=self.fake_settings(),
        ), mock.patch(
            "app.agent.llm.factory.create_extraction_provider", return_value=llm
        ):
            return self.client.post(
                f"/users/{self.test_user_id}/documents/{doc_id}/extract"
            )

    def test_upload_extract_list_round_trip(self):
        doc_id = self._upload_png()
        llm = CannedLLM([valid_draft_json()])

        resp = self._extract(doc_id, llm)
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertFalse(body["cached"])
        self.assertEqual(body["status"], "draft")
        self.assertEqual(body["document_id"], doc_id)
        self.assertEqual(body["extracted_by"], "test/extraction-model")
        self.assertEqual(body["draft"]["issuer"], "City Power")
        self.assertTrue(body["validation_result"]["passed"])

        # Second run on the same bytes: served from cache, no new model call.
        resp = self._extract(doc_id, llm)
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertTrue(resp.json()["cached"])
        self.assertEqual(len(llm.received), 1)

        with mock.patch(
            "app.services.document_service.get_settings",
            return_value=self.fake_settings(),
        ):
            listing = self.client.get(
                f"/users/{self.test_user_id}/documents/{doc_id}/extractions"
            )
        self.assertEqual(listing.status_code, 200, listing.text)
        self.assertEqual(len(listing.json()), 1)
        self.assertEqual(listing.json()[0]["id"], body["id"])

    def test_extract_without_llm_is_503(self):
        doc_id = self._upload_png()
        resp = self._extract(doc_id, None)
        self.assertEqual(resp.status_code, 503)
        self.assertIn("No LLM provider", resp.json()["detail"])

    def test_extract_unknown_document_is_404(self):
        resp = self._extract("no-such-doc", CannedLLM([]))
        self.assertEqual(resp.status_code, 404)

    def test_extract_invalid_draft_is_400(self):
        doc_id = self._upload_png()
        llm = CannedLLM(["garbage one", "garbage two"])
        resp = self._extract(doc_id, llm)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("draft", resp.json()["detail"])

    def test_extract_transport_error_is_502(self):
        from app.agent.llm.omniroute_provider import OmniRouteError

        doc_id = self._upload_png()

        class ExplodingLLM:
            model = "test/exploding"

            def generate(self, messages):
                raise OmniRouteError("connection refused (scrubbed)")

        resp = self._extract(doc_id, ExplodingLLM())
        self.assertEqual(resp.status_code, 502)

    def test_extraction_not_visible_across_users(self):
        doc_id = self._upload_png()
        resp = self._extract(doc_id, CannedLLM([valid_draft_json()]))
        self.assertEqual(resp.status_code, 200, resp.text)
        extraction_id = resp.json()["id"]

        # A different user sees nothing: no documents, no extractions.
        other = self.make_user()
        with mock.patch(
            "app.services.document_service.get_settings",
            return_value=self.fake_settings(),
        ):
            listing = self.client.get(
                f"/users/{other['id']}/documents/{doc_id}/extractions"
            )
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.json(), [])


if __name__ == "__main__":
    unittest.main()

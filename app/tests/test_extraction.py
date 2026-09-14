"""Extraction pipeline tests (Phase 4: multimodal document understanding).

Every test runs against a canned LLM — no socket may open (the module-level
guard makes an accidental real call impossible), no credential is needed, and
the pipeline's deterministic parts (validation, parsing, caching, storage,
user isolation) are asserted directly.
"""
import json
import socket
import tempfile
import unittest
import uuid
from datetime import date
from typing import Any, List, Optional

from sqlalchemy import delete

from app.db.models import AuditEvent, Base, User
from app.tests.test_api import ApiTestBase
from app.services.user_service import create_user
from app.services.document_service import register_document
from app.services.extraction_service import (
    ExtractionDraft,
    ExtractionServiceError,
    build_extraction_prompt,
    extract_document,
    extract_json_object,
    get_extraction,
    list_extractions,
    parse_draft,
    validate_draft,
)

_ORIGINAL_SOCKET = socket.socket


def _no_network(*args, **kwargs):
    raise AssertionError(
        "network access is not permitted in the offline extraction test suite; "
        "use a canned LLM (or the opt-in live smoke test) for real calls."
    )


def setUpModule():
    socket.socket = _no_network


def tearDownModule():
    socket.socket = _ORIGINAL_SOCKET


# --- canned LLM ---------------------------------------------------------------

class CannedExtractionLLM:
    """Returns scripted replies; records every message list it receives."""

    def __init__(self, replies: List[str], model: str = "test/extraction-model"):
        self.replies = list(replies)
        self.received: List[List[Any]] = []
        self.model = model

    def generate(self, messages):
        self.received.append(list(messages))
        if not self.replies:
            raise AssertionError("canned LLM ran out of scripted replies")
        return _TextResponse(self.replies.pop(0))


class _TextResponse:
    def __init__(self, text: Optional[str]):
        self.text_content = text
        self.tool_calls = None


def valid_draft_json(**overrides: Any) -> str:
    payload = {
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
        "field_confidence": {"total_amount_minor": 0.95, "issue_date": 0.9},
        "notes": "",
    }
    payload.update(overrides)
    return json.dumps(payload)


class ExtractionTestBase(ApiTestBase):
    """ApiTestBase plus a per-test user, temp storage root, and cleanup."""

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
        self.today = date(2026, 9, 14)
        self.storage_root = tempfile.mkdtemp(prefix="extraction-test-")
        self.as_of = self.today

    def tearDown(self):
        with self.SessionFactory() as db:
            for table in reversed(Base.metadata.sorted_tables):
                if "user_id" in table.c:
                    db.execute(delete(table).where(table.c.user_id == self.test_user_id))
            db.execute(delete(User).where(User.id == self.test_user_id))
            db.commit()

    def register_png(self, content: bytes = b"\x89PNG fake-image-bytes", **kwargs):
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            doc = register_document(
                db,
                user,
                filename=kwargs.pop("filename", "bill.png"),
                mime_type=kwargs.pop("mime_type", "image/png"),
                content=content,
                storage_root=self.storage_root,
                **kwargs,
            )
            db.commit()
            return doc.id, doc.content_hash


# --- schema + validation ------------------------------------------------------

class TestDraftSchema(unittest.TestCase):

    def test_valid_draft_round_trips(self):
        draft = parse_draft(valid_draft_json())
        self.assertEqual(draft.document_type, "bill")
        self.assertEqual(draft.total_amount_minor, 8500)
        self.assertEqual(draft.currency, "USD")
        self.assertEqual(len(draft.line_items), 2)

    def test_bad_document_type_rejected(self):
        with self.assertRaises(ValueError):
            parse_draft(valid_draft_json(document_type="tree"))

    def test_fractional_amount_rejected_not_rounded(self):
        # A fractional amount must fail schema validation (and trigger a retry),
        # never be silently rounded into money.
        with self.assertRaises(ValueError):
            parse_draft(valid_draft_json(total_amount_minor=85.5))

    def test_lowercase_currency_normalized(self):
        draft = parse_draft(valid_draft_json(currency="usd"))
        self.assertEqual(draft.currency, "USD")

    def test_bad_currency_rejected(self):
        with self.assertRaises(ValueError):
            parse_draft(valid_draft_json(currency="dollars"))

    def test_nonpositive_amount_rejected(self):
        with self.assertRaises(ValueError):
            parse_draft(valid_draft_json(total_amount_minor=0))

    def test_confidence_bounds_enforced(self):
        with self.assertRaises(ValueError):
            parse_draft(valid_draft_json(field_confidence={"total_amount_minor": 1.5}))


class TestJsonExtraction(unittest.TestCase):
    """The tolerant parser handles the ways models wrap JSON."""

    def test_plain_json(self):
        self.assertEqual(extract_json_object('{"a": 1}'), {"a": 1})

    def test_fenced_json(self):
        text = 'Here you go:\n```json\n{"a": 1}\n```\nHope that helps!'
        self.assertEqual(extract_json_object(text), {"a": 1})

    def test_prose_wrapped_json(self):
        text = 'The extracted fields are {"a": 1, "b": {"c": 2}} as requested.'
        self.assertEqual(extract_json_object(text), {"a": 1, "b": {"c": 2}})

    def test_no_json_raises(self):
        with self.assertRaises(ValueError):
            extract_json_object("I could not read the document.")

    def test_unbalanced_json_raises(self):
        with self.assertRaises(ValueError):
            extract_json_object('{"a": 1')

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            extract_json_object("   ")


class TestDraftValidation(unittest.TestCase):
    """Deterministic validation findings (ARCHITECTURE.md §5.1)."""

    def setUp(self):
        self.as_of = date(2026, 9, 14)

    def _draft(self, **overrides: Any) -> ExtractionDraft:
        payload = json.loads(valid_draft_json(**overrides))
        return ExtractionDraft.model_validate(payload)

    def test_clean_draft_passes(self):
        result = validate_draft(self._draft(), self.as_of)
        self.assertTrue(result["passed"])
        self.assertEqual(result["findings"], [])

    def test_line_sum_mismatch_is_a_finding(self):
        result = validate_draft(
            self._draft(total_amount_minor=9000), self.as_of
        )
        self.assertFalse(result["passed"])
        codes = [f["code"] for f in result["findings"]]
        self.assertIn("line_items_sum_mismatch", codes)

    def test_due_before_issue_is_a_finding(self):
        result = validate_draft(
            self._draft(issue_date="2026-08-20", due_date="2026-08-01"), self.as_of
        )
        codes = [f["code"] for f in result["findings"]]
        self.assertIn("date_order_invalid", codes)

    def test_implausibly_old_date_is_a_finding(self):
        result = validate_draft(self._draft(issue_date="1971-01-01"), self.as_of)
        codes = [f["code"] for f in result["findings"]]
        self.assertIn("date_out_of_bounds", codes)

    def test_far_future_date_is_a_finding(self):
        result = validate_draft(self._draft(issue_date="2040-01-01"), self.as_of)
        codes = [f["code"] for f in result["findings"]]
        self.assertIn("date_out_of_bounds", codes)


# --- the pipeline -------------------------------------------------------------

class TestExtractDocument(ExtractionTestBase):

    def test_successful_extraction_stores_draft(self):
        doc_id, _ = self.register_png()
        llm = CannedExtractionLLM([valid_draft_json()])
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            extraction, cached = extract_document(
                db, user, document_id=doc_id, llm=llm, as_of=self.as_of,
                storage_root=self.storage_root,
            )
            db.commit()

            self.assertFalse(cached)
            self.assertEqual(extraction.status, "draft")
            self.assertEqual(extraction.document_id, doc_id)
            self.assertEqual(extraction.extracted_by, "test/extraction-model")
            self.assertEqual(extraction.draft["total_amount_minor"], 8500)
            self.assertTrue(extraction.validation_result["passed"])

            # The LLM saw exactly one user turn with the image attached.
            messages = llm.received[0]
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0].role.value, "user")
            self.assertEqual(messages[0].attachments[0].media_type, "image/png")
            self.assertIn(b"PNG fake-image-bytes", __import__("base64").b64decode(
                messages[0].attachments[0].data_b64
            ))
            # The prompt carries the schema (structured-output enforcement).
            self.assertIn("JSON object", messages[0].content)
            self.assertIn("minor units", messages[0].content)

    def test_cached_by_content_hash(self):
        """Re-extracting identical bytes reuses the stored draft, no model call."""
        doc_id, _ = self.register_png()
        llm = CannedExtractionLLM([valid_draft_json()])
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            first, cached_first = extract_document(
                db, user, document_id=doc_id, llm=llm, as_of=self.as_of,
                storage_root=self.storage_root,
            )
            db.commit()

        # A second *document row* with the same bytes must hit the cache.
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            doc2, _ = self.register_png(content=b"\x89PNG fake-image-bytes")
            db.commit()
            second, cached_second = extract_document(
                db, user, document_id=doc2, llm=llm, as_of=self.as_of,
                storage_root=self.storage_root,
            )
            db.commit()

        self.assertFalse(cached_first)
        self.assertTrue(cached_second)
        self.assertEqual(first.id, second.id)
        self.assertEqual(len(llm.received), 1)  # only the first extraction called

    def test_retry_on_invalid_then_success(self):
        doc_id, _ = self.register_png()
        llm = CannedExtractionLLM(
            ["Sorry, I cannot read that.", valid_draft_json()]
        )
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            extraction, cached = extract_document(
                db, user, document_id=doc_id, llm=llm, as_of=self.as_of,
                storage_root=self.storage_root,
            )
            db.commit()

        self.assertFalse(cached)
        self.assertEqual(extraction.status, "draft")
        # Two model calls: the failed attempt and the corrective retry.
        self.assertEqual(len(llm.received), 2)
        # The retry saw the failed reply plus the corrective instruction.
        retry_messages = llm.received[1]
        self.assertEqual([m.role.value for m in retry_messages], ["user", "assistant", "user"])
        self.assertIn("not valid", retry_messages[2].content)

    def test_two_failures_raise_domain_error(self):
        doc_id, _ = self.register_png()
        llm = CannedExtractionLLM(["garbage one", "garbage two"])
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            with self.assertRaises(ExtractionServiceError) as cm:
                extract_document(
                    db, user, document_id=doc_id, llm=llm, as_of=self.as_of,
                    storage_root=self.storage_root,
                )
            db.rollback()
        self.assertEqual(cm.exception.code, "invalid_draft")
        self.assertEqual(len(llm.received), 2)  # no third attempt

    def test_unsupported_mime_rejected(self):
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            doc = register_document(
                db, user,
                filename="payslip.pdf",
                mime_type="application/pdf",
                content=b"%PDF-1.4 fake",
                storage_root=self.storage_root,
            )
            db.commit()
            llm = CannedExtractionLLM([valid_draft_json()])
            with self.assertRaises(ExtractionServiceError) as cm:
                extract_document(
                    db, user, document_id=doc.id, llm=llm, as_of=self.as_of,
                    storage_root=self.storage_root,
                )
            db.rollback()
        self.assertEqual(cm.exception.code, "extraction_not_supported")
        self.assertEqual(len(llm.received), 0)  # refused before any model call

    def test_missing_content_raises(self):
        """A metadata row whose bytes are gone fails honestly, not with a guess."""
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            doc = register_document(
                db, user,
                filename="ghost.png",
                mime_type="image/png",
                content=b"\x89PNG ghost",
                storage_root=self.storage_root,
            )
            db.commit()

        # Simulate storage loss.
        import os
        os.remove(os.path.join(self.storage_root, user.id, doc.content_hash))

        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            llm = CannedExtractionLLM([valid_draft_json()])
            with self.assertRaises(ExtractionServiceError) as cm:
                extract_document(
                    db, user, document_id=doc.id, llm=llm, as_of=self.as_of,
                    storage_root=self.storage_root,
                )
            db.rollback()
        self.assertEqual(cm.exception.code, "document_content_missing")

    def test_unknown_document_is_not_found(self):
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            llm = CannedExtractionLLM([])
            with self.assertRaises(ExtractionServiceError) as cm:
                extract_document(
                    db, user, document_id="no-such-doc", llm=llm, as_of=self.as_of,
                    storage_root=self.storage_root,
                )
            db.rollback()
        self.assertEqual(cm.exception.code, "document_not_found")

    def test_user_isolation(self):
        """User B cannot extract (or read) user A's document."""
        with self.SessionFactory() as db:
            user_a = db.get(User, self.test_user_id)
            other = create_user(
                db,
                email=f"{uuid.uuid4().hex[:10]}@example.com",
                home_currency="USD",
                current_balance_minor=10000,
            )
            db.commit()

            doc = register_document(
                db, user_a,
                filename="private.png",
                mime_type="image/png",
                content=b"\x89PNG private",
                storage_root=self.storage_root,
            )
            llm = CannedExtractionLLM([valid_draft_json()])
            extraction, _ = extract_document(
                db, user_a, document_id=doc.id, llm=llm, as_of=self.as_of,
                storage_root=self.storage_root,
            )
            db.commit()

            # The other user cannot extract the document...
            with self.assertRaises(ExtractionServiceError) as cm:
                extract_document(
                    db, other, document_id=doc.id, llm=llm, as_of=self.as_of,
                    storage_root=self.storage_root,
                )
            db.rollback()
            self.assertEqual(cm.exception.code, "document_not_found")

            # ...nor read the extraction it produced.
            with self.assertRaises(ExtractionServiceError) as cm:
                get_extraction(db, other, extraction.id)
            self.assertEqual(cm.exception.code, "extraction_not_found")

            # Owner still can.
            own = get_extraction(db, user_a, extraction.id)
            self.assertEqual(own.id, extraction.id)

            # Cleanup the extra user's rows.
            for table in reversed(Base.metadata.sorted_tables):
                if "user_id" in table.c:
                    db.execute(delete(table).where(table.c.user_id == other.id))
            db.execute(delete(User).where(User.id == other.id))
            db.commit()

    def test_validation_findings_stored_with_draft(self):
        """A schema-valid but semantically odd draft is stored with its findings."""
        doc_id, _ = self.register_png()
        odd = valid_draft_json(
            total_amount_minor=9000,               # line sum mismatch
            issue_date="2026-08-20",
            due_date="2026-08-01",                 # due before issue
        )
        llm = CannedExtractionLLM([odd])
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            extraction, _ = extract_document(
                db, user, document_id=doc_id, llm=llm, as_of=self.as_of,
                storage_root=self.storage_root,
            )
            db.commit()

        self.assertEqual(extraction.status, "draft")
        self.assertFalse(extraction.validation_result["passed"])
        codes = [f["code"] for f in extraction.validation_result["findings"]]
        self.assertIn("line_items_sum_mismatch", codes)
        self.assertIn("date_order_invalid", codes)

    def test_extraction_audited(self):
        doc_id, _ = self.register_png()
        llm = CannedExtractionLLM([valid_draft_json()])
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            extract_document(
                db, user, document_id=doc_id, llm=llm, as_of=self.as_of,
                storage_root=self.storage_root,
            )
            db.commit()
            events = db.scalars(
                select_audit_events(db, self.test_user_id, "document_extraction_saved")
            ).all()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].payload["document_id"], doc_id)


def select_audit_events(db, user_id: str, event_type: str):
    from sqlalchemy import select
    return select(AuditEvent).where(
        AuditEvent.user_id == user_id, AuditEvent.event_type == event_type
    )


class TestListExtractions(ExtractionTestBase):

    def test_listing_scoped_to_document_and_user(self):
        doc_id, _ = self.register_png()
        other_id, _ = self.register_png(content=b"\x89PNG different", filename="other.png")
        llm = CannedExtractionLLM([valid_draft_json(), valid_draft_json(total_amount_minor=100)])
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            extract_document(
                db, user, document_id=doc_id, llm=llm, as_of=self.as_of,
                storage_root=self.storage_root,
            )
            db.commit()
            extract_document(
                db, user, document_id=other_id, llm=llm, as_of=self.as_of,
                storage_root=self.storage_root,
            )
            db.commit()

            only_doc = list_extractions(db, user, document_id=doc_id)
            self.assertEqual(len(only_doc), 1)
            self.assertEqual(only_doc[0].document_id, doc_id)

            everything = list_extractions(db, user)
            self.assertEqual(len(everything), 2)


# --- prompt -------------------------------------------------------------------

class TestExtractionPrompt(unittest.TestCase):

    def test_prompt_carries_schema_and_rules(self):
        prompt = build_extraction_prompt()
        self.assertIn("total_amount_minor", prompt)
        self.assertIn("minor units", prompt)
        self.assertIn("document_type", prompt)
        self.assertIn("payslip", prompt)


if __name__ == "__main__":
    unittest.main()

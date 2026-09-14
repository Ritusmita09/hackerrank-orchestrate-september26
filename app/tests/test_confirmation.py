"""Confirmation-flow tests (Phase 4: user confirmation of inferred facts).

The confirmation path is fully deterministic — no LLM is constructed, no
network is needed — so drafts and pattern proposals are created directly as
rows and confirmed through the service, the API, and the agent tool.

What is asserted, in priority order:

* nothing reaches the ledger without a confirmation (and exactly one event
  per confirmation),
* provenance, status, and confidence of ingested events,
* user isolation and refusal codes (invalid status, currency, direction),
* pattern proposals only influence state reconstruction after confirmation.
"""
import unittest
import uuid
from datetime import date, timedelta

from pydantic import ValidationError
from sqlalchemy import delete, select

from app.db.models import (
    AuditEvent, Base, Document, DocumentExtraction, RecurringPattern, User,
)
from app.tests.test_api import ApiTestBase
from app.services.user_service import create_user
from app.services.extraction_service import (
    ExtractionDraft,
    draft_to_json,
    get_cached_extraction,
)
from app.services.confirmation_service import (
    ConfirmationServiceError,
    confirm_extraction,
    confirm_pattern,
    reject_extraction,
    reject_pattern,
)
from app.services.financial_state_service import get_financial_state
from app.services.pattern_service import (
    get_confirmed_patterns,
    list_patterns,
)
from app.agent.tools import registry


AS_OF = date(2026, 9, 14)


class ConfirmationTestBase(ApiTestBase):
    """Per-test user + row factories; every test starts from a clean slate."""

    def setUp(self):
        super().setUp()
        with self.SessionFactory() as db:
            user = create_user(
                db,
                email=f"{uuid.uuid4().hex[:10]}@example.com",
                home_currency="USD",
                current_balance_minor=100_000,
            )
            db.commit()
            self.test_user_id = user.id

    def tearDown(self):
        with self.SessionFactory() as db:
            for table in reversed(Base.metadata.sorted_tables):
                if "user_id" in table.c:
                    db.execute(delete(table).where(table.c.user_id == self.test_user_id))
            db.execute(delete(User).where(User.id == self.test_user_id))
            db.commit()

    def _make_user(self, **overrides) -> str:
        with self.SessionFactory() as db:
            other = create_user(
                db,
                email=f"{uuid.uuid4().hex[:10]}@example.com",
                home_currency=overrides.pop("home_currency", "USD"),
                current_balance_minor=100_000,
                **overrides,
            )
            db.commit()
            return other.id

    def make_document(self, **overrides) -> str:
        """A document row (metadata only — confirmation never reads bytes)."""
        with self.SessionFactory() as db:
            doc = Document(
                user_id=self.test_user_id,
                kind=overrides.pop("kind", "bill"),
                filename=overrides.pop("filename", "bill.png"),
                content_hash=overrides.pop(
                    "content_hash", uuid.uuid4().hex
                ),
                mime_type=overrides.pop("mime_type", "image/png"),
                size_bytes=overrides.pop("size_bytes", 10),
                **overrides,
            )
            db.add(doc)
            db.commit()
            return doc.id

    def make_extraction(self, draft: ExtractionDraft, *, document_id: str,
                        status: str = "draft") -> str:
        with self.SessionFactory() as db:
            extraction = DocumentExtraction(
                document_id=document_id,
                draft=draft_to_json(draft),
                validation_result={"passed": True, "findings": []},
                status=status,
                extracted_by="test/model",
            )
            db.add(extraction)
            db.commit()
            return extraction.id

    def make_pattern(self, *, user_id: str = None, user_confirmed: bool = False,
                     active: bool = True) -> str:
        with self.SessionFactory() as db:
            row = RecurringPattern(
                user_id=user_id or self.test_user_id,
                stream_key=f"stream-{uuid.uuid4().hex[:8]}",
                category="utilities",
                direction="debit",
                period_type="monthly",
                period_days_of_month=[1],
                typical_amount_minor=8_500,
                estimator="median",
                currency="USD",
                last_seen=AS_OF - timedelta(days=30),
                evidence_event_ids=[],
                confidence=0.9,
                user_confirmed=user_confirmed,
                active=active,
                description="City Power",
            )
            db.add(row)
            db.commit()
            return row.id

    def bill_draft(self, **overrides) -> ExtractionDraft:
        payload = {
            "document_type": "bill",
            "issuer": "City Power",
            "issue_date": "2026-08-01",
            "due_date": "2026-08-20",
            "currency": "USD",
            "total_amount_minor": 8_500,
            "line_items": [
                {"description": "Electricity", "amount_minor": 7_000},
                {"description": "Service fee", "amount_minor": 1_500},
            ],
            "field_confidence": {
                "total_amount_minor": 0.95, "issue_date": 0.9, "due_date": 0.9,
            },
        }
        payload.update(overrides)
        return ExtractionDraft.model_validate(payload)


# --- extraction confirmation (service) ---------------------------------------

class TestExtractionConfirmation(ConfirmationTestBase):

    def test_confirm_bill_ingests_debit_on_due_date(self):
        doc_id = self.make_document()
        extraction_id = self.make_extraction(self.bill_draft(), document_id=doc_id)

        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            extraction, txn = confirm_extraction(
                db, user, extraction_id=extraction_id, as_of=AS_OF
            )
            db.commit()

            self.assertEqual(extraction.status, "confirmed")
            self.assertIsNotNone(extraction.confirmed_at)

            self.assertEqual(txn.date, date(2026, 8, 20))   # due date
            self.assertEqual(txn.direction, "debit")        # bills are money out
            self.assertEqual(txn.amount_minor, 8_500)
            self.assertEqual(txn.currency, "USD")
            self.assertEqual(txn.status, "settled")         # due_date <= as_of
            self.assertEqual(txn.category, "bill")
            self.assertEqual(txn.description, "City Power bill")
            self.assertEqual(txn.source_type, "document")
            self.assertEqual(txn.source_id, doc_id)         # provenance doc:<id>
            self.assertEqual(txn.confidence, 0.95)          # total's own confidence
            self.assertTrue(txn.confirmed_by_user)

    def test_confirm_future_bill_is_scheduled(self):
        doc_id = self.make_document()
        extraction_id = self.make_extraction(
            self.bill_draft(due_date="2026-10-01"), document_id=doc_id
        )
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            _, txn = confirm_extraction(
                db, user, extraction_id=extraction_id, as_of=AS_OF
            )
            db.commit()
            self.assertEqual(txn.status, "scheduled")       # future flow

    def test_confirm_payslip_ingests_credit_on_issue_date(self):
        doc_id = self.make_document(kind="payslip")
        draft = self.bill_draft(
            document_type="payslip",
            issuer="Acme Corp",
            issue_date="2026-08-31",
            due_date=None,
            total_amount_minor=320_000,
            field_confidence={"total_amount_minor": 0.99},
        )
        extraction_id = self.make_extraction(draft, document_id=doc_id)

        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            _, txn = confirm_extraction(
                db, user, extraction_id=extraction_id, as_of=AS_OF
            )
            db.commit()
            self.assertEqual(txn.direction, "credit")       # payslip is money in
            self.assertEqual(txn.date, date(2026, 8, 31))   # issue date
            self.assertEqual(txn.category, "payslip")
            self.assertEqual(txn.description, "Acme Corp payslip")
            self.assertEqual(txn.confidence, 0.99)

    def test_confirm_with_user_corrections(self):
        doc_id = self.make_document()
        extraction_id = self.make_extraction(self.bill_draft(), document_id=doc_id)

        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            _, txn = confirm_extraction(
                db, user, extraction_id=extraction_id, as_of=AS_OF,
                direction="credit",
                category="utilities",
                description="City Power - August",
                event_date=date(2026, 9, 1),
            )
            db.commit()
            self.assertEqual(txn.direction, "credit")
            self.assertEqual(txn.category, "utilities")
            self.assertEqual(txn.description, "City Power - August")
            self.assertEqual(txn.date, date(2026, 9, 1))
            # The amount always comes from the validated draft.
            self.assertEqual(txn.amount_minor, 8_500)

    def test_confirm_statement_requires_direction(self):
        doc_id = self.make_document(kind="statement")
        draft = self.bill_draft(document_type="statement", issuer="Bank")
        extraction_id = self.make_extraction(draft, document_id=doc_id)

        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            with self.assertRaises(ConfirmationServiceError) as ctx:
                confirm_extraction(db, user, extraction_id=extraction_id, as_of=AS_OF)
            self.assertEqual(ctx.exception.code, "direction_required")

    def test_confirm_rejects_bad_direction(self):
        doc_id = self.make_document()
        extraction_id = self.make_extraction(self.bill_draft(), document_id=doc_id)
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            with self.assertRaises(ConfirmationServiceError) as ctx:
                confirm_extraction(
                    db, user, extraction_id=extraction_id, as_of=AS_OF,
                    direction="sideways",
                )
            self.assertEqual(ctx.exception.code, "invalid_direction")

    def test_confirm_refuses_foreign_currency(self):
        doc_id = self.make_document()
        draft = self.bill_draft(currency="EUR")
        extraction_id = self.make_extraction(draft, document_id=doc_id)
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            with self.assertRaises(ConfirmationServiceError) as ctx:
                confirm_extraction(db, user, extraction_id=extraction_id, as_of=AS_OF)
            self.assertEqual(ctx.exception.code, "currency_mismatch")
            self.assertIn("never converted", ctx.exception.message)

    def test_confirm_requires_a_date(self):
        doc_id = self.make_document(kind="other")
        draft = self.bill_draft(
            document_type="other", issue_date=None, due_date=None
        )
        extraction_id = self.make_extraction(draft, document_id=doc_id)
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            with self.assertRaises(ConfirmationServiceError) as ctx:
                confirm_extraction(
                    db, user, extraction_id=extraction_id, as_of=AS_OF,
                    direction="debit",
                )
            self.assertEqual(ctx.exception.code, "date_required")

    def test_confirm_only_once(self):
        doc_id = self.make_document()
        extraction_id = self.make_extraction(self.bill_draft(), document_id=doc_id)
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            confirm_extraction(db, user, extraction_id=extraction_id, as_of=AS_OF)
            db.commit()
            with self.assertRaises(ConfirmationServiceError) as ctx:
                confirm_extraction(db, user, extraction_id=extraction_id, as_of=AS_OF)
            self.assertEqual(ctx.exception.code, "invalid_status")

    def test_confirm_rejected_draft_refused(self):
        doc_id = self.make_document()
        extraction_id = self.make_extraction(
            self.bill_draft(), document_id=doc_id, status="rejected"
        )
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            with self.assertRaises(ConfirmationServiceError) as ctx:
                confirm_extraction(db, user, extraction_id=extraction_id, as_of=AS_OF)
            self.assertEqual(ctx.exception.code, "invalid_status")

    def test_confirm_unknown_extraction(self):
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            with self.assertRaises(ConfirmationServiceError) as ctx:
                confirm_extraction(db, user, extraction_id="nope", as_of=AS_OF)
            self.assertEqual(ctx.exception.code, "extraction_not_found")

    def test_confirm_is_user_scoped(self):
        other_id = self._make_user()
        doc_id = self.make_document()
        extraction_id = self.make_extraction(self.bill_draft(), document_id=doc_id)
        with self.SessionFactory() as db:
            other = db.get(User, other_id)
            with self.assertRaises(ConfirmationServiceError) as ctx:
                confirm_extraction(db, other, extraction_id=extraction_id, as_of=AS_OF)
            self.assertEqual(ctx.exception.code, "extraction_not_found")

    def test_confirm_duplicate_values_refused(self):
        # Two different documents carrying identical draft values: the first
        # confirmation ingests; the second must not create a second event.
        first_doc = self.make_document()
        second_doc = self.make_document()
        first = self.make_extraction(self.bill_draft(), document_id=first_doc)
        second = self.make_extraction(self.bill_draft(), document_id=second_doc)

        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            confirm_extraction(db, user, extraction_id=first, as_of=AS_OF)
            db.commit()
            with self.assertRaises(ConfirmationServiceError) as ctx:
                confirm_extraction(db, user, extraction_id=second, as_of=AS_OF)
            self.assertEqual(ctx.exception.code, "duplicate_event")

    def test_confidence_falls_back_to_mean(self):
        doc_id = self.make_document()
        draft = self.bill_draft(
            field_confidence={"issue_date": 0.8, "due_date": 0.6}
        )
        extraction_id = self.make_extraction(draft, document_id=doc_id)
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            _, txn = confirm_extraction(
                db, user, extraction_id=extraction_id, as_of=AS_OF
            )
            db.commit()
            self.assertAlmostEqual(txn.confidence, 0.7)

    def test_audit_event_recorded(self):
        doc_id = self.make_document()
        extraction_id = self.make_extraction(self.bill_draft(), document_id=doc_id)
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            _, txn = confirm_extraction(
                db, user, extraction_id=extraction_id, as_of=AS_OF
            )
            db.commit()

            events = db.scalars(
                select(AuditEvent)
                .where(AuditEvent.user_id == user.id)
                .where(AuditEvent.event_type == "extraction_confirmed")
            ).all()
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].payload["extraction_id"], extraction_id)
            self.assertEqual(events[0].payload["transaction_id"], txn.id)
            self.assertEqual(events[0].payload["document_id"], doc_id)

    def test_confirmed_event_reaches_financial_state(self):
        doc_id = self.make_document()
        extraction_id = self.make_extraction(self.bill_draft(), document_id=doc_id)
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            confirm_extraction(db, user, extraction_id=extraction_id, as_of=AS_OF)
            db.commit()

            state = get_financial_state(db, user, as_of=AS_OF)
            self.assertGreaterEqual(len(state.history), 1)


# --- extraction rejection (service) ------------------------------------------

class TestExtractionRejection(ConfirmationTestBase):

    def test_reject_draft(self):
        doc_id = self.make_document()
        extraction_id = self.make_extraction(self.bill_draft(), document_id=doc_id)
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            extraction = reject_extraction(db, user, extraction_id=extraction_id)
            db.commit()

            self.assertEqual(extraction.status, "rejected")
            events = db.scalars(
                select(AuditEvent)
                .where(AuditEvent.user_id == user.id)
                .where(AuditEvent.event_type == "extraction_rejected")
            ).all()
            self.assertEqual(len(events), 1)

    def test_reject_is_idempotent(self):
        doc_id = self.make_document()
        extraction_id = self.make_extraction(self.bill_draft(), document_id=doc_id)
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            reject_extraction(db, user, extraction_id=extraction_id)
            db.commit()
            extraction = reject_extraction(db, user, extraction_id=extraction_id)
            db.commit()
            self.assertEqual(extraction.status, "rejected")
            events = db.scalars(
                select(AuditEvent)
                .where(AuditEvent.user_id == user.id)
                .where(AuditEvent.event_type == "extraction_rejected")
            ).all()
            self.assertEqual(len(events), 1)   # no duplicate audit spam

    def test_reject_confirmed_refused(self):
        doc_id = self.make_document()
        extraction_id = self.make_extraction(self.bill_draft(), document_id=doc_id)
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            confirm_extraction(db, user, extraction_id=extraction_id, as_of=AS_OF)
            db.commit()
            with self.assertRaises(ConfirmationServiceError) as ctx:
                reject_extraction(db, user, extraction_id=extraction_id)
            self.assertEqual(ctx.exception.code, "invalid_status")

    def test_rejected_draft_is_reextracted_next_run(self):
        # The content-hash cache reuses only draft/confirmed rows: a rejected
        # draft forces a fresh model call on the next extraction.
        doc_id = self.make_document()
        extraction_id = self.make_extraction(self.bill_draft(), document_id=doc_id)
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            reject_extraction(db, user, extraction_id=extraction_id)
            db.commit()
            doc = db.get(Document, doc_id)
            self.assertIsNone(get_cached_extraction(db, user, doc.content_hash))

    def test_reject_unknown_extraction(self):
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            with self.assertRaises(ConfirmationServiceError) as ctx:
                reject_extraction(db, user, extraction_id="nope")
            self.assertEqual(ctx.exception.code, "extraction_not_found")


# --- pattern confirmation (service) ------------------------------------------

class TestPatternConfirmation(ConfirmationTestBase):

    def test_confirm_pattern_sets_flag_and_audits(self):
        pattern_id = self.make_pattern()
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            row = confirm_pattern(db, user, pattern_id=pattern_id)
            db.commit()

            self.assertTrue(row.user_confirmed)
            self.assertTrue(row.active)
            events = db.scalars(
                select(AuditEvent)
                .where(AuditEvent.user_id == user.id)
                .where(AuditEvent.event_type == "pattern_confirmed")
            ).all()
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].payload["pattern_id"], pattern_id)

    def test_confirm_pattern_idempotent(self):
        pattern_id = self.make_pattern()
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            confirm_pattern(db, user, pattern_id=pattern_id)
            db.commit()
            row = confirm_pattern(db, user, pattern_id=pattern_id)
            db.commit()
            self.assertTrue(row.user_confirmed)
            events = db.scalars(
                select(AuditEvent)
                .where(AuditEvent.user_id == user.id)
                .where(AuditEvent.event_type == "pattern_confirmed")
            ).all()
            self.assertEqual(len(events), 1)

    def test_confirm_inactive_pattern_refused(self):
        pattern_id = self.make_pattern(active=False)
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            with self.assertRaises(ConfirmationServiceError) as ctx:
                confirm_pattern(db, user, pattern_id=pattern_id)
            self.assertEqual(ctx.exception.code, "pattern_inactive")

    def test_reject_pattern_deactivates(self):
        pattern_id = self.make_pattern()
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            row = reject_pattern(db, user, pattern_id=pattern_id)
            db.commit()

            self.assertFalse(row.active)
            self.assertEqual(
                [], [p.pattern_id for p in get_confirmed_patterns(db, user)]
            )
            # Hidden from the default listing, visible with include_inactive.
            self.assertEqual(
                [], [p.id for p in list_patterns(db, user)]
            )
            self.assertEqual(
                [pattern_id], [p.id for p in list_patterns(db, user, include_inactive=True)]
            )

    def test_reject_pattern_idempotent(self):
        pattern_id = self.make_pattern()
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            reject_pattern(db, user, pattern_id=pattern_id)
            db.commit()
            row = reject_pattern(db, user, pattern_id=pattern_id)
            db.commit()
            self.assertFalse(row.active)
            events = db.scalars(
                select(AuditEvent)
                .where(AuditEvent.user_id == user.id)
                .where(AuditEvent.event_type == "pattern_rejected")
            ).all()
            self.assertEqual(len(events), 1)

    def test_pattern_errors_are_user_scoped(self):
        other_id = self._make_user()
        pattern_id = self.make_pattern()
        with self.SessionFactory() as db:
            other = db.get(User, other_id)
            for call in (
                lambda: confirm_pattern(db, other, pattern_id=pattern_id),
                lambda: reject_pattern(db, other, pattern_id=pattern_id),
            ):
                with self.assertRaises(ConfirmationServiceError) as ctx:
                    call()
                self.assertEqual(ctx.exception.code, "pattern_not_found")

    def test_unknown_pattern(self):
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            with self.assertRaises(ConfirmationServiceError) as ctx:
                confirm_pattern(db, user, pattern_id="nope")
            self.assertEqual(ctx.exception.code, "pattern_not_found")

    def test_only_confirmed_patterns_reach_state(self):
        # The core invariant of the confirmation gate: an unconfirmed proposal
        # must not influence state reconstruction.
        proposal_id = self.make_pattern()
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            state = get_financial_state(db, user, as_of=AS_OF)
            self.assertEqual(len(state.patterns), 0)

            confirm_pattern(db, user, pattern_id=proposal_id)
            db.commit()
            state = get_financial_state(db, user, as_of=AS_OF)
            self.assertEqual(len(state.patterns), 1)

            reject_pattern(db, user, pattern_id=proposal_id)
            db.commit()
            state = get_financial_state(db, user, as_of=AS_OF)
            self.assertEqual(len(state.patterns), 0)

    def test_list_patterns_filters(self):
        proposal_id = self.make_pattern()
        confirmed_id = self.make_pattern(user_confirmed=True)
        rejected_id = self.make_pattern(active=False)

        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            all_active = {p.id for p in list_patterns(db, user)}
            self.assertEqual(all_active, {proposal_id, confirmed_id})

            proposals = {p.id for p in list_patterns(db, user, confirmed=False)}
            self.assertEqual(proposals, {proposal_id})

            confirmed = {p.id for p in list_patterns(db, user, confirmed=True)}
            self.assertEqual(confirmed, {confirmed_id})

            everything = {p.id for p in list_patterns(db, user, include_inactive=True)}
            self.assertEqual(everything, {proposal_id, confirmed_id, rejected_id})


# --- API routes ---------------------------------------------------------------

class TestConfirmationApi(ConfirmationTestBase):

    def test_confirm_route(self):
        doc_id = self.make_document()
        extraction_id = self.make_extraction(self.bill_draft(), document_id=doc_id)

        resp = self.client.post(
            f"/users/{self.test_user_id}/documents/{doc_id}/extractions/"
            f"{extraction_id}/confirm"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["status"], "confirmed")
        self.assertTrue(body["transaction_id"])
        self.assertEqual(body["direction"], "debit")
        self.assertEqual(body["amount_minor"], 8_500)
        self.assertEqual(body["date"], "2026-08-20")
        self.assertEqual(body["transaction_status"], "settled")

        # A second confirmation is refused — one ingest per draft.
        resp_again = self.client.post(
            f"/users/{self.test_user_id}/documents/{doc_id}/extractions/"
            f"{extraction_id}/confirm"
        )
        self.assertEqual(resp_again.status_code, 400)
        self.assertIn("only drafts can be confirmed", resp_again.json()["detail"])

    def test_confirm_route_with_corrections(self):
        doc_id = self.make_document()
        extraction_id = self.make_extraction(self.bill_draft(), document_id=doc_id)

        resp = self.client.post(
            f"/users/{self.test_user_id}/documents/{doc_id}/extractions/"
            f"{extraction_id}/confirm",
            json={
                "category": "utilities",
                "event_date": "2026-09-01",
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["date"], "2026-09-01")

    def test_confirm_route_wrong_document_404(self):
        doc_id = self.make_document()
        other_doc_id = self.make_document()
        extraction_id = self.make_extraction(self.bill_draft(), document_id=doc_id)

        resp = self.client.post(
            f"/users/{self.test_user_id}/documents/{other_doc_id}/extractions/"
            f"{extraction_id}/confirm"
        )
        self.assertEqual(resp.status_code, 404)

    def test_confirm_route_unknown_extraction_404(self):
        doc_id = self.make_document()
        resp = self.client.post(
            f"/users/{self.test_user_id}/documents/{doc_id}/extractions/"
            f"does-not-exist/confirm"
        )
        self.assertEqual(resp.status_code, 404)

    def test_reject_route(self):
        doc_id = self.make_document()
        extraction_id = self.make_extraction(self.bill_draft(), document_id=doc_id)

        resp = self.client.post(
            f"/users/{self.test_user_id}/documents/{doc_id}/extractions/"
            f"{extraction_id}/reject"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["status"], "rejected")

    def test_patterns_routes(self):
        proposal_id = self.make_pattern()

        listing = self.client.get(f"/users/{self.test_user_id}/patterns")
        self.assertEqual(listing.status_code, 200, listing.text)
        rows = listing.json()
        self.assertEqual([r["id"] for r in rows], [proposal_id])
        self.assertFalse(rows[0]["user_confirmed"])

        confirmed = self.client.post(
            f"/users/{self.test_user_id}/patterns/{proposal_id}/confirm"
        )
        self.assertEqual(confirmed.status_code, 200, confirmed.text)
        self.assertTrue(confirmed.json()["user_confirmed"])

        # The proposal queue is empty now.
        pending = self.client.get(
            f"/users/{self.test_user_id}/patterns", params={"confirmed": "false"}
        )
        self.assertEqual(pending.status_code, 200)
        self.assertEqual(pending.json(), [])

        rejected = self.client.post(
            f"/users/{self.test_user_id}/patterns/{proposal_id}/reject"
        )
        self.assertEqual(rejected.status_code, 200, rejected.text)
        self.assertFalse(rejected.json()["active"])

        # Default listing hides the retired pattern.
        after = self.client.get(f"/users/{self.test_user_id}/patterns")
        self.assertEqual(after.json(), [])

    def test_pattern_routes_unknown_404(self):
        resp = self.client.post(
            f"/users/{self.test_user_id}/patterns/does-not-exist/confirm"
        )
        self.assertEqual(resp.status_code, 404)


# --- agent tool ---------------------------------------------------------------

class TestConfirmExtractionTool(ConfirmationTestBase):

    def test_tool_registered_with_schema(self):
        schemas = registry.get_all_schemas()
        tool = next(s for s in schemas if s["name"] == "confirm_extraction")
        self.assertTrue(tool["description"])
        self.assertIn("extraction_id", tool["input_schema"]["properties"])
        self.assertIn(
            "extraction_id", tool["input_schema"].get("required", [])
        )

    def test_tool_confirms_and_ingests(self):
        doc_id = self.make_document()
        extraction_id = self.make_extraction(self.bill_draft(), document_id=doc_id)

        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            result = registry.execute(
                "confirm_extraction",
                db=db,
                user=user,
                args={"extraction_id": extraction_id},
                as_of=AS_OF,
            )
            db.commit()

        self.assertTrue(result["success"])
        self.assertTrue(result["transaction_id"])
        self.assertEqual(result["direction"], "debit")
        self.assertEqual(result["amount_minor"], 8_500)
        self.assertIn(f"document:{doc_id}", result["note"])

    def test_tool_reports_domain_errors(self):
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            result = registry.execute(
                "confirm_extraction",
                db=db,
                user=user,
                args={"extraction_id": "nope"},
                as_of=AS_OF,
            )
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "extraction_not_found")

    def test_tool_requires_extraction_id(self):
        with self.SessionFactory() as db:
            user = db.get(User, self.test_user_id)
            with self.assertRaises(ValidationError):
                registry.execute(
                    "confirm_extraction",
                    db=db,
                    user=user,
                    args={},
                    as_of=AS_OF,
                )


if __name__ == "__main__":
    unittest.main()

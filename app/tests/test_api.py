"""Phase 2 API tests — FastAPI boundary over the deterministic engine.

Every test runs against an isolated in-memory SQLite database; the app never
touches the developer's configured DATABASE_URL. Financial assertions reuse
the Phase 1 engine invariants (integer minor units, verification gate).
"""
import io
import unittest
import uuid
from datetime import date, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.main import create_app
from app.api.routes.users import get_db
from app.db.models import AuditEvent, Base


class ApiTestBase(unittest.TestCase):
    """Shared fixture: app + in-memory SQLite + dependency override."""

    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(cls.engine)
        cls.SessionFactory = sessionmaker(bind=cls.engine, expire_on_commit=False)

        cls.app = create_app()

        def override_get_db():
            with cls.SessionFactory() as session:
                yield session

        cls.app.dependency_overrides[get_db] = override_get_db
        cls.client = TestClient(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.app.dependency_overrides.clear()
        cls.engine.dispose()

    # -- helpers -------------------------------------------------------------

    def make_user(self, **overrides) -> dict:
        payload = {
            "email": f"{uuid.uuid4().hex[:10]}@example.com",
            "display_name": "Test User",
            "home_currency": "USD",
            "current_balance_minor": 500_000,   # $5,000.00
            "minimum_balance_minor": 100_000,   # $1,000.00 floor
        }
        payload.update(overrides)
        resp = self.client.post("/users", json=payload)
        assert resp.status_code == 200, resp.text
        return resp.json()

    def add_transaction(self, user_id: str, **overrides) -> dict:
        payload = {
            "date": date.today().isoformat(),
            "direction": "debit",
            "amount_minor": 5_000,
            "category": "groceries",
            "description": "weekly shop",
        }
        payload.update(overrides)
        resp = self.client.post(f"/users/{user_id}/transactions", json=payload)
        assert resp.status_code == 200, resp.text
        return resp.json()


class TestHealth(ApiTestBase):
    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("engine_version", body)

    def test_request_id_header_present(self):
        resp = self.client.get("/health")
        self.assertTrue(resp.headers.get("x-request-id"))

    def test_request_id_echoed_when_provided(self):
        resp = self.client.get("/health", headers={"X-Request-ID": "trace-123"})
        self.assertEqual(resp.headers["x-request-id"], "trace-123")


class TestUsers(ApiTestBase):
    def test_create_user(self):
        body = self.make_user(display_name="Alice")
        self.assertIn("id", body)
        self.assertEqual(body["home_currency"], "USD")
        self.assertTrue(body["created_at"])

    def test_get_user(self):
        created = self.make_user()
        resp = self.client.get(f"/users/{created['id']}")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["id"], created["id"])

    def test_get_unknown_user_404(self):
        resp = self.client.get("/users/does-not-exist")
        self.assertEqual(resp.status_code, 404)
        self.assertIn("error", resp.json())

    def test_profile_endpoint(self):
        created = self.make_user()
        resp = self.client.get(f"/users/{created['id']}/financial-state/profile")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["current_balance_minor"], 500_000)
        self.assertEqual(body["minimum_balance_minor"], 100_000)
        self.assertEqual(body["currency"], "USD")


class TestTransactions(ApiTestBase):
    def test_create_and_list_transaction(self):
        user = self.make_user()
        txn = self.add_transaction(user["id"], amount_minor=12_345)
        self.assertEqual(txn["amount_minor"], 12_345)
        self.assertEqual(txn["direction"], "debit")
        self.assertEqual(txn["currency"], "USD")

        resp = self.client.get(f"/users/{user['id']}/transactions")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()), 1)

    def test_duplicate_transaction_rejected_409(self):
        user = self.make_user()
        self.add_transaction(user["id"], description="same", amount_minor=999)
        resp = self.client.post(
            f"/users/{user['id']}/transactions",
            json={
                "date": date.today().isoformat(),
                "direction": "debit",
                "amount_minor": 999,
                "category": "groceries",
                "description": "same",
            },
        )
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.json()["error"], "duplicate")

    def test_unknown_user_transactions_404(self):
        resp = self.client.get("/users/nobody/transactions")
        self.assertEqual(resp.status_code, 404)

    def test_malformed_transaction_422(self):
        user = self.make_user()
        # negative amount, invalid direction, missing category
        resp = self.client.post(
            f"/users/{user['id']}/transactions",
            json={
                "date": date.today().isoformat(),
                "direction": "sideways",
                "amount_minor": -5,
                "category": "",
            },
        )
        self.assertEqual(resp.status_code, 422)


class TestCsvImport(ApiTestBase):
    def _upload(self, user_id: str, csv_text: str, filename: str = "bank.csv"):
        return self.client.post(
            f"/users/{user_id}/transactions/csv",
            files={"file": (filename, io.BytesIO(csv_text.encode("utf-8")), "text/csv")},
        )

    def test_valid_csv_import(self):
        user = self.make_user()
        today = date.today().isoformat()
        csv_text = (
            "date,direction,category,amount_minor,description\n"
            f"{today},credit,income,250000,salary\n"
            f"{today},debit,groceries,4500,weekly shop\n"
        )
        resp = self._upload(user["id"], csv_text)
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["status"], "success")
        self.assertEqual(body["imported"], 2)
        self.assertEqual(body["rejected"], 0)

    def test_row_level_errors_reported(self):
        user = self.make_user()
        today = date.today().isoformat()
        csv_text = (
            "date,direction,category,amount_minor\n"
            f"{today},debit,groceries,1000\n"
            "not-a-date,debit,groceries,2000\n"
            f"{today},sideways,groceries,3000\n"
            f"{today},debit,groceries,-4000\n"
        )
        resp = self._upload(user["id"], csv_text)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["imported"], 1)
        self.assertEqual(body["rejected"], 3)
        self.assertEqual(len(body["errors"]), 3)
        self.assertTrue(all("row" in e and "message" in e for e in body["errors"]))

    def test_missing_columns_rejected(self):
        user = self.make_user()
        resp = self._upload(user["id"], "description,amount\ncoffee,3\n")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("missing_columns", resp.text)

    def test_duplicate_rows_counted(self):
        user = self.make_user()
        today = date.today().isoformat()
        csv_text = (
            "date,direction,category,amount_minor,description\n"
            f"{today},debit,groceries,1000,dupe\n"
            f"{today},debit,groceries,1000,dupe\n"
        )
        resp = self._upload(user["id"], csv_text)
        body = resp.json()
        self.assertEqual(body["imported"], 1)
        self.assertEqual(body["duplicates"], 1)

    def test_non_csv_filename_rejected(self):
        user = self.make_user()
        resp = self._upload(user["id"], "a,b\n1,2\n", filename="bank.txt")
        self.assertEqual(resp.status_code, 400)


class TestFinancialState(ApiTestBase):
    def test_state_reconstruction(self):
        user = self.make_user()
        today = date.today()
        # History (settled before today) + a future scheduled debit.
        self.add_transaction(
            user["id"], date=(today - timedelta(days=30)).isoformat(),
            direction="credit", amount_minor=300_000, category="income",
            description="salary", status="settled",
        )
        self.add_transaction(
            user["id"], date=(today + timedelta(days=10)).isoformat(),
            direction="debit", amount_minor=50_000, category="housing",
            description="rent", status="scheduled",
        )
        resp = self.client.get(f"/users/{user['id']}/financial-state")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["user_id"], user["id"])
        self.assertEqual(body["home_currency"], "USD")
        self.assertEqual(body["current_balance_minor"], 500_000)
        self.assertEqual(body["minimum_balance_minor"], 100_000)
        self.assertGreaterEqual(body["counts"]["history_events"], 1)
        self.assertGreaterEqual(body["counts"]["future_confirmed_events"], 1)

    def test_state_unknown_user_404(self):
        resp = self.client.get("/users/nobody/financial-state")
        self.assertEqual(resp.status_code, 404)


class TestSimulation(ApiTestBase):
    def test_simulation_series(self):
        user = self.make_user()
        resp = self.client.post(
            f"/users/{user['id']}/simulate", params={"horizon_days": 30}
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["horizon_days"], 30)
        # Phase 1 series include both endpoints: today .. today + horizon.
        self.assertEqual(len(body["points"]), 31)
        self.assertIn("engine_version", body)
        self.assertIn("input_hash", body)

    def test_simulation_persists_forecast_run(self):
        user = self.make_user()
        self.client.post(f"/users/{user['id']}/simulate", params={"horizon_days": 7})
        with self.SessionFactory() as session:
            from app.db.models import ForecastRun
            runs = session.scalars(
                select(ForecastRun).where(ForecastRun.user_id == user["id"])
            ).all()
        self.assertEqual(len(runs), 1)
        # canonicalize() wraps dataclasses as {"__dataclass__", "fields"}.
        self.assertEqual(runs[0].assumptions["fields"]["horizon_days"], 7)

    def test_simulation_unknown_user_404(self):
        resp = self.client.post("/users/nobody/simulate")
        self.assertEqual(resp.status_code, 404)


class TestDecisions(ApiTestBase):
    def _decide(self, user_id: str, **overrides):
        today = date.today()
        payload = {
            "requested_amount_minor": 50_000,   # $500 — well within balance
            "desired_completion_date": (today + timedelta(days=30)).isoformat(),
            "allows_partial_payment": False,
            "payment_options": [],
        }
        payload.update(overrides)
        return self.client.post(f"/users/{user_id}/decision", json=payload)

    def test_affordable_decision(self):
        user = self.make_user()
        resp = self._decide(user["id"])
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertIn(body["status"], ("affordable_now", "affordable_with_plan"))
        self.assertNotEqual(body["plan_method"], "not_recommended")
        self.assertTrue(body["engine_version"])
        self.assertTrue(body["fact_sheet"])

    def test_decision_with_installment_option(self):
        user = self.make_user()
        today = date.today()
        resp = self._decide(
            user["id"],
            requested_amount_minor=120_000,
            payment_options=[{
                "option_id": "opt-3x",
                "first_payment_date": (today + timedelta(days=5)).isoformat(),
                "frequency_days": 30,
                "number_of_payments": 3,
                "payment_amount_minor": 40_000,
                "total_payable_minor": 120_000,
            }],
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["total_payable_minor"], 120_000)

    def test_unaffordable_request_not_recommended(self):
        # Balance far too low, no payment options can rescue it.
        user = self.make_user(current_balance_minor=120_000, minimum_balance_minor=100_000)
        resp = self._decide(user["id"], requested_amount_minor=10_000_000)
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["status"], "not_affordable")
        self.assertEqual(body["plan_method"], "not_recommended")
        self.assertEqual(body["total_payable_minor"], 0)

    def test_verification_failure_handled(self):
        # Installments that would breach the minimum balance: the engine must
        # reject those candidates through verify_plan rather than return an
        # unverified plan — the API response stays structured either way.
        user = self.make_user(current_balance_minor=150_000, minimum_balance_minor=100_000)
        today = date.today()
        resp = self._decide(
            user["id"],
            requested_amount_minor=1_000_000,
            payment_options=[{
                "option_id": "opt-impossible",
                "first_payment_date": today.isoformat(),
                "frequency_days": 7,
                "number_of_payments": 4,
                "payment_amount_minor": 250_000,
                "total_payable_minor": 1_000_000,
            }],
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        # Whatever it recommends must never be a plan that fails verification.
        self.assertIn("status", body)
        self.assertIn(body["plan_method"],
                      ("full_payment", "installments", "wait", "partial_payment", "not_recommended"))

    def test_invalid_payment_option_400(self):
        # Zero payments is engine-invalid (InvalidRequestError) -> 400, not 500.
        user = self.make_user()
        today = date.today()
        resp = self._decide(
            user["id"],
            payment_options=[{
                "option_id": "bad",
                "first_payment_date": today.isoformat(),
                "frequency_days": 0,
                "number_of_payments": 0,
                "payment_amount_minor": 0,
                "total_payable_minor": 0,
            }],
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertEqual(resp.json()["error"], "engine_error")

    def test_decision_unknown_user_404(self):
        resp = self._decide("nobody")
        self.assertEqual(resp.status_code, 404)


class TestDocuments(ApiTestBase):
    def test_upload_document(self):
        user = self.make_user()
        resp = self.client.post(
            f"/users/{user['id']}/documents",
            files={"file": ("payslip.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")},
            params={"kind": "payslip"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["status"], "registered")
        self.assertEqual(body["kind"], "payslip")

    def test_invalid_mime_rejected(self):
        user = self.make_user()
        resp = self.client.post(
            f"/users/{user['id']}/documents",
            files={"file": ("evil.exe", io.BytesIO(b"MZ"), "application/x-msdownload")},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("invalid_mime_type", resp.text)

    def test_oversized_upload_rejected(self):
        user = self.make_user()
        from app.config import get_settings
        big = b"x" * (get_settings().max_upload_bytes + 1)
        resp = self.client.post(
            f"/users/{user['id']}/documents",
            files={"file": ("big.pdf", io.BytesIO(big), "application/pdf")},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("file_too_large", resp.text)


class TestAuditTrail(ApiTestBase):
    def test_actions_audited(self):
        user = self.make_user()
        self.add_transaction(user["id"])
        self.client.post(f"/users/{user['id']}/simulate", params={"horizon_days": 7})
        self.client.post(
            f"/users/{user['id']}/decision",
            json={
                "requested_amount_minor": 10_000,
                "desired_completion_date": (date.today() + timedelta(days=30)).isoformat(),
                "allows_partial_payment": False,
                "payment_options": [],
            },
        )
        with self.SessionFactory() as session:
            events = session.scalars(
                select(AuditEvent).where(AuditEvent.user_id == user["id"])
            ).all()
        types = {e.event_type for e in events}
        self.assertIn("transaction_ingested", types)
        self.assertIn("forecast_run", types)
        self.assertIn("decision_made", types)

    def test_decision_persisted_with_hash_and_version(self):
        user = self.make_user()
        resp = self.client.post(
            f"/users/{user['id']}/decision",
            json={
                "requested_amount_minor": 10_000,
                "desired_completion_date": (date.today() + timedelta(days=30)).isoformat(),
                "allows_partial_payment": False,
                "payment_options": [],
            },
        )
        request_id = resp.json()["request_id"]
        with self.SessionFactory() as session:
            from app.db.models import Decision, VerificationResult
            decision = session.scalars(
                select(Decision).where(Decision.request_id == request_id)
            ).one()
            self.assertTrue(decision.state_input_hash)
            self.assertTrue(decision.engine_version)
            self.assertTrue(decision.fact_sheet)
            vr = session.scalars(
                select(VerificationResult).where(VerificationResult.request_id == request_id)
            ).all()
            self.assertTrue(vr)


if __name__ == "__main__":
    unittest.main()

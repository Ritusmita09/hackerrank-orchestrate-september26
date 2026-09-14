"""Audit trail service — every consequential action leaves an append-only record."""
from __future__ import annotations

from sqlalchemy.orm import Session

from ..db.models import AuditEvent


def record_audit_event(
    db: Session,
    *,
    event_type: str,
    user_id: str | None = None,
    request_id: str | None = None,
    payload: dict | None = None,
) -> AuditEvent:
    """Append one audit event. Payloads must stay small and non-sensitive."""
    event = AuditEvent(
        user_id=user_id,
        request_id=request_id,
        event_type=event_type,
        payload=payload or {},
    )
    db.add(event)
    return event

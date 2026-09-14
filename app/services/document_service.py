"""Documents and messages — bounded untrusted input for future extraction.

Documents are untrusted input (SECURITY_AND_SAFETY.md §5): bytes live outside
the database (under the configured storage root), the database holds metadata
plus the content hash used for dedup and extraction caching, and every payload
is size- and MIME-bounded before it is accepted.
"""
from __future__ import annotations

import hashlib
import os

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db.models import Document, Message, User
from .audit_service import record_audit_event


class InputServiceError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _storage_root(storage_root: str | None) -> str:
    return storage_root if storage_root is not None else get_settings().storage_dir


def write_document_content(
    user_id: str, content_hash: str, content: bytes, *, storage_root: str | None = None
) -> str:
    """Persist document bytes under the storage root, content-hash addressed.

    Returns the path written. The hash-named layout makes identical uploads
    collapse onto one file and keeps provenance verifiable: the stored bytes
    always hash to the name of the file they live in.
    """
    root = _storage_root(storage_root)
    user_dir = os.path.join(root, user_id)
    os.makedirs(user_dir, exist_ok=True)
    path = os.path.join(user_dir, content_hash)
    with open(path, "wb") as fh:
        fh.write(content)
    return path


def read_document_content(
    user_id: str, content_hash: str, *, storage_root: str | None = None
) -> bytes:
    """Read back document bytes; raises InputServiceError when they are gone."""
    path = os.path.join(_storage_root(storage_root), user_id, content_hash)
    if not os.path.isfile(path):
        raise InputServiceError(
            "document_content_missing",
            "the document's stored content could not be found on disk",
        )
    with open(path, "rb") as fh:
        return fh.read()


def register_document(
    db: Session,
    user: User,
    *,
    filename: str,
    mime_type: str,
    content: bytes,
    kind: str = "other",
    storage_root: str | None = None,
) -> Document:
    """Validate limits and store document metadata (content goes to disk/S3)."""
    settings = get_settings()

    if len(content) > settings.max_upload_bytes:
        raise InputServiceError(
            "file_too_large",
            f"File size {len(content)} exceeds {settings.max_upload_bytes} bytes",
        )
    if mime_type not in settings.allowed_mime_types:
        raise InputServiceError(
            "invalid_mime_type",
            f"MIME type {mime_type!r} is not allowed (allowed: {', '.join(settings.allowed_mime_types)})",
        )

    content_hash = _hash_bytes(content)

    # Bytes go to storage, not the DB (ARCHITECTURE.md §7). The metadata row
    # keeps only the hash, so extraction can verify and re-read them later.
    storage_path = write_document_content(
        user.id, content_hash, content, storage_root=storage_root
    )

    doc = Document(
        user_id=user.id,
        kind=kind,
        filename=filename,
        content_hash=content_hash,
        mime_type=mime_type,
        size_bytes=len(content),
        storage_path=storage_path,
    )
    db.add(doc)
    record_audit_event(
        db, event_type="document_registered",
        user_id=user.id,
        payload={"filename": filename, "mime_type": mime_type, "size_bytes": len(content)},
    )
    db.flush()
    return doc


def get_document(db: Session, user_id: str, document_id: str) -> Document:
    doc = db.get(Document, document_id)
    if doc is None or doc.user_id != user_id:
        raise InputServiceError("document_not_found", "document not found")
    return doc


def receive_message(db: Session, user: User, raw_text: str) -> Message:
    """Store raw text input.

    Phase 3 will interpret this to construct amendment records or respond.
    """
    settings = get_settings()
    raw_text = raw_text.strip()
    if len(raw_text) > settings.max_message_chars:
        raise InputServiceError(
            "message_too_long",
            f"message exceeds maximum length of {settings.max_message_chars} characters",
        )
    if not raw_text:
        raise InputServiceError("empty_message", "message text is empty")

    msg = Message(user_id=user.id, raw_text=raw_text)
    db.add(msg)
    record_audit_event(
        db, event_type="message_received",
        user_id=user.id, payload={"length": len(raw_text)},
    )
    db.flush()
    return msg

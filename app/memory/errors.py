"""Memory errors — the same ``code``/``message`` shape services use elsewhere."""
from __future__ import annotations


class MemoryServiceError(Exception):
    """A memory operation the store refuses to perform.

    ``code`` is a stable, machine-readable identifier (``invalid_goal``,
    ``secret_rejected``, ``not_found``, ``reserved_key``...). ``message`` is
    safe to show a user and never contains stored values.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

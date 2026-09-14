"""Credential guard — memory must never become a place secrets are stored.

SECURITY_AND_SAFETY.md §4: no credentials, API keys, or tokens belong in the
repository, the code, or the database. Memory is user-authored free text, so
it is exactly the kind of store a user (or a confused agent) might paste a
password into. These writes are refused, and the refusal message never echoes
the value it rejected.

Two independent checks:

* **Key names** that name a credential (``password``, ``api_key``, ``token``...)
  are refused regardless of value — storing one is never intended.
* **Value shapes** that look like credentials (known provider prefixes such as
  ``sk-``, or a long unbroken token-like blob) are refused regardless of key.

Both are deliberately conservative: ordinary sentences, dates, and category
names must pass untouched.
"""
from __future__ import annotations

import re
from typing import Any

from .errors import MemoryServiceError

#: Key names that denote a credential. Matched as substrings on a normalized
#: (lowercased, separator-stripped) key so ``api_key``, ``apiKey`` and
#: ``api-key`` all match.
_CREDENTIAL_KEY_TERMS = (
    "password",
    "passwd",
    "passphrase",
    "secret",
    "token",
    "apikey",
    "api_key",
    "accesskey",
    "privatekey",
    "credential",
    "clientsecret",
    "authtoken",
    "bearer",
    "sessionid",
    "cookie",
    "cvv",
    "cvc",
    "pin",
    "seedphrase",
    "mnemonic",
    "recoverycode",
)

#: Value prefixes that identify a credential from a known issuer.
_CREDENTIAL_PREFIXES = (
    "sk-",
    "sk_live_",
    "sk_test_",
    "rk_live_",
    "ghp_",
    "gho_",
    "ghs_",
    "github_pat_",
    "xoxb-",
    "xoxp-",
    "xoxa-",
    "AKIA",
    "ASIA",
    "AIza",
    "ya29.",
    "glpat-",
    "Bearer ",
    "Basic ",
    "eyJ",  # JWT
    "-----BEGIN",
)

#: A long, unbroken, token-like blob: no whitespace, 40+ chars of the
#: base64/hex/url-safe charset. Sentences (which contain spaces) never match.
_TOKEN_BLOB = re.compile(r"^[A-Za-z0-9_\-+/=.]{40,}$")

_NON_ALNUM = re.compile(r"[^a-z0-9]")


def _normalize_key(key: str) -> str:
    return _NON_ALNUM.sub("", str(key).lower())


def is_credential_key(key: str) -> bool:
    """True when a key name denotes a credential."""
    normalized = _normalize_key(key)
    return any(term.replace("_", "") in normalized for term in _CREDENTIAL_KEY_TERMS)


def looks_like_credential_value(value: str) -> bool:
    """True when a string is shaped like a credential or API key."""
    text = value.strip()
    if not text:
        return False
    if any(text.startswith(prefix) for prefix in _CREDENTIAL_PREFIXES):
        return True
    return bool(_TOKEN_BLOB.match(text))


def find_credential(path: str, value: Any) -> str | None:
    """Return the path of the first credential-shaped value found, else None.

    Walks mappings and sequences so a credential nested inside a preference
    object is caught just the same.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            key_path = f"{path}.{key}" if path else str(key)
            if is_credential_key(str(key)):
                return key_path
            found = find_credential(key_path, item)
            if found:
                return found
        return None

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = find_credential(f"{path}[{index}]", item)
            if found:
                return found
        return None

    if isinstance(value, str) and looks_like_credential_value(value):
        return path or "(value)"
    return None


def assert_storable(key: str, value: Any) -> None:
    """Raise if ``key``/``value`` would put a credential into memory.

    The error message names the offending *key path* only — never the value —
    so a rejected secret cannot leak through an error message, a log line, or
    a test failure.
    """
    if is_credential_key(key):
        raise MemoryServiceError(
            "secret_rejected",
            f"refusing to store {key!r}: memory never holds credentials, "
            "API keys, or tokens (SECURITY_AND_SAFETY.md §4).",
        )

    path = find_credential(key, value)
    if path is not None:
        raise MemoryServiceError(
            "secret_rejected",
            f"refusing to store {key!r}: the value at {path!r} looks like a "
            "credential, API key, or token. Memory never holds secrets.",
        )

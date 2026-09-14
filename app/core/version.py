"""Engine versioning and deterministic input hashing.

Every result the engine produces (forecasts, verifications, recommendations)
carries ``ENGINE_VERSION`` plus a hash of the exact inputs that produced it.
Together these make any past decision reproducible and auditable: re-running
the same engine version over the same hashed inputs yields the same output.

The hash is computed over a canonical JSON serialization, so it is stable
across processes, platforms, and Python versions (unlike ``hash()``).
"""
from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from datetime import date
from decimal import Decimal
from typing import Any

#: Version of the deterministic engine. Bump on ANY behavioral change,
#: however small: a recommendation produced by "1.0.0" must be replayable
#: by "1.0.0" and only "1.0.0".
ENGINE_VERSION = "1.0.0"


def canonicalize(obj: Any) -> Any:
    """Convert an object into a JSON-serializable canonical form.

    Handles the engine's vocabulary: dataclasses (including ``Money``),
    enums, dates, Decimals, tuples/lists, and sets (order-insensitive).

    Raises:
        TypeError: if the object contains something non-canonicalizable.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        # Floats never represent money in this engine; they only appear as
        # confidence scores. ``repr`` round-trips exactly.
        return {"__float__": repr(obj)}
    if isinstance(obj, Decimal):
        return {"__decimal__": str(obj)}
    if isinstance(obj, date):
        return {"__date__": obj.isoformat()}
    if isinstance(obj, enum.Enum):
        return {"__enum__": type(obj).__name__, "value": canonicalize(obj.value)}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            "__dataclass__": type(obj).__qualname__,
            "fields": {
                f.name: canonicalize(getattr(obj, f.name))
                for f in dataclasses.fields(obj)
            },
        }
    if isinstance(obj, (tuple, list)):
        return [canonicalize(item) for item in obj]
    if isinstance(obj, (frozenset, set)):
        return {"__set__": sorted(canonical_json(item) for item in obj)}
    if isinstance(obj, dict):
        return {str(key): canonicalize(value) for key, value in obj.items()}
    raise TypeError(f"Cannot canonicalize object of type {type(obj).__name__!r}")


def canonical_json(obj: Any) -> str:
    """Return the canonical JSON string for ``obj`` (sorted keys, tight separators)."""
    return json.dumps(canonicalize(obj), sort_keys=True, separators=(",", ":"))


def input_hash(*objs: Any) -> str:
    """Return a deterministic SHA-256 hash of the engine version plus each input.

    The engine version is folded into the hash so that the same inputs under a
    different engine version produce a different hash — making it impossible to
    accidentally attribute an old result to a new engine.
    """
    digest = hashlib.sha256()
    digest.update(ENGINE_VERSION.encode("utf-8"))
    for obj in objs:
        digest.update(b"\x00")
        digest.update(canonical_json(obj).encode("utf-8"))
    return digest.hexdigest()

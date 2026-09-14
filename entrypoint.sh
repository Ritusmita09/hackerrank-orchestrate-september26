#!/bin/sh
# Deployment entrypoint: migrate first, then serve.
#
# Alembic is the authoritative schema migration mechanism. This script runs
# every pending migration BEFORE the API accepts traffic, so a fresh
# deployment can never serve against a missing or stale schema.
#
# Local development uses the same sequence:
#   python -m alembic upgrade head
#   python -m uvicorn app.api.main:create_app --factory --reload
set -e

echo "==> Applying database migrations (alembic upgrade head)"
python -m alembic upgrade head

echo "==> Starting API"
exec python -m uvicorn app.api.main:create_app --factory --host 0.0.0.0 --port 8000

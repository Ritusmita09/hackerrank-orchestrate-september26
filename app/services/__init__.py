"""Service layer — all financial business logic lives here, never in routes.

Services translate persistence rows into the engine's typed domain models,
call the deterministic engine, and persist the audit trail. Routes only
parse/validate input, call a service, and shape the response.
"""

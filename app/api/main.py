"""FastAPI application factory and security middleware."""
import logging
import uuid
from contextvars import ContextVar

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from ..core.models import EngineError
from ..services.document_service import InputServiceError
from ..services.transaction_service import TransactionServiceError
from ..services.user_service import UserServiceError
from .routes import (
    decisions,
    documents,
    finances,
    health,
    simulation,
    transactions,
    users,
)

logger = logging.getLogger(__name__)

# Request ID context for logs
request_id_ctx_var: ContextVar[str] = ContextVar("request_id", default="")


def create_app() -> FastAPI:
    app = FastAPI(title="AI Financial Decision Agent - Core Engine API")

    app.include_router(health.router)
    app.include_router(users.router, prefix="/users", tags=["users"])

    # Financial sub-resources for the user
    app.include_router(finances.router, prefix="/users/{user_id}", tags=["finances"])
    app.include_router(transactions.router, prefix="/users/{user_id}/transactions", tags=["transactions"])
    app.include_router(documents.router, prefix="/users/{user_id}/documents", tags=["documents"])
    app.include_router(decisions.router, prefix="/users/{user_id}/decision", tags=["decisions"])
    app.include_router(simulation.router, prefix="/users/{user_id}/simulate", tags=["simulation"])

    @app.middleware("http")
    async def add_request_id_logger(request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        token = request_id_ctx_var.set(req_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        request_id_ctx_var.reset(token)
        return response

    # Map our domain errors directly to standard JSON responses so internal
    # exception traces never leak to the caller.
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("unhandled_error", extra={"error": str(exc)})
        return JSONResponse(status_code=500, content={"error": "internal_server_error"})

    @app.exception_handler(SQLAlchemyError)
    async def sql_exception_handler(request: Request, exc: SQLAlchemyError):
        logger.error("database_error", extra={"error": str(exc)})
        return JSONResponse(status_code=500, content={"error": "database_error"})

    @app.exception_handler(EngineError)
    async def engine_exception_handler(request: Request, exc: EngineError):
        return JSONResponse(status_code=400, content={"error": "engine_error", "message": str(exc)})

    @app.exception_handler(UserServiceError)
    async def user_exception_handler(request: Request, exc: UserServiceError):
        status = 404 if "not_found" in exc.code else 400
        return JSONResponse(status_code=status, content={"error": exc.code, "message": exc.message})

    @app.exception_handler(TransactionServiceError)
    async def txn_exception_handler(request: Request, exc: TransactionServiceError):
        status = 409 if exc.code == "duplicate" else 400
        return JSONResponse(status_code=status, content={"error": exc.code, "message": exc.message})

    @app.exception_handler(InputServiceError)
    async def doc_exception_handler(request: Request, exc: InputServiceError):
        status = 404 if exc.code == "document_not_found" else 400
        return JSONResponse(status_code=status, content={"error": exc.code, "message": exc.message})

    # (CsvImportError and PatternServiceError caught similarly in routes if needed)

    return app

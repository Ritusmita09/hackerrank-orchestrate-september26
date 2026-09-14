from fastapi import APIRouter
from pydantic import BaseModel

from ...core.version import ENGINE_VERSION

router = APIRouter()

class HealthResponse(BaseModel):
    status: str
    engine_version: str

@router.get("/health", response_model=HealthResponse)
def get_health():
    """Service health and current deterministic-engine version."""
    return HealthResponse(status="ok", engine_version=ENGINE_VERSION)

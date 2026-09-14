from typing import Iterator

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...db.base import build_engine, build_session_factory
from ...config import get_settings
from ..schemas import UserCreate, UserResponse

router = APIRouter()

_session_factory = None

def get_db() -> Iterator[Session]:
    """Dependency: independent Session per request."""
    # We initialize the DB lazily so importing the app doesn't connect.
    global _session_factory
    if _session_factory is None:
        settings = get_settings()
        engine = build_engine(settings.database_url, echo=(settings.log_level == "DEBUG"))
        _session_factory = build_session_factory(engine)

    with _session_factory() as session:
        yield session


@router.post("", response_model=UserResponse)
def create_user(req: UserCreate, db: Session = Depends(get_db)):
    """Create a new user with an initial financial profile."""
    from ...services.user_service import create_user as svc_create
    user = svc_create(
        db,
        email=req.email,
        display_name=req.display_name,
        home_currency=req.home_currency,
        current_balance_minor=req.current_balance_minor,
        minimum_balance_minor=req.minimum_balance_minor,
    )
    db.commit()
    return user


@router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: str, db: Session = Depends(get_db)):
    from ...services.user_service import get_user as svc_get
    user = svc_get(db, user_id)
    return user

from fastapi import APIRouter, Depends

from app.api.deps import get_services
from app.api.schemas import HealthOut
from app.services.container import Services

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
def health(svc: Services = Depends(get_services)):
    qdrant_ok = svc.store.healthy()
    llm_ok = svc.llm is not None  # configured; we don't spend free-tier quota on a ping
    return HealthOut(status="ok" if qdrant_ok and llm_ok else "degraded", api=True, qdrant=qdrant_ok,
                     llm=llm_ok, retrieval_mode=svc.settings.retrieval_mode)

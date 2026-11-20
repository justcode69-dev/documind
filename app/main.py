"""FastAPI entrypoint:  uvicorn app.main:app --reload"""

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, documents, health
from app.api.security import RateLimiter, require_api_key
from app.config import Settings, get_settings
from app.services.container import Services, build_services

# Export .env into os.environ too, so LangSmith (LANGSMITH_*) picks up tracing settings.
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = services or build_services(settings)
        app.state.rate_limiter = RateLimiter(settings.rate_limit_per_minute)
        yield

    app = FastAPI(title="DocuMind", version="1.0.0", lifespan=lifespan,
                  description="RAG document Q&A with page-level citations.")
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"],
                       allow_headers=["*"])
    app.dependency_overrides[get_settings] = lambda: settings
    auth = [Depends(require_api_key)]
    app.include_router(health.router)
    app.include_router(documents.router, dependencies=auth)
    app.include_router(chat.router, dependencies=auth)
    return app


app = create_app()

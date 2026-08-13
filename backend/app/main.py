from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.infrastructure.db.migrate import run_migrations_on_startup

configure_logging()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # See migrate.py - applies any pending Alembic migration automatically so
    # a schema change never again sits unapplied in production until someone
    # remembers to run it by hand against Supabase.
    run_migrations_on_startup()
    yield


app = FastAPI(
    title="QuantumAlpha API",
    description="Backend for personal quantitative portfolio management.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    return {"status": "ok", "environment": settings.environment}

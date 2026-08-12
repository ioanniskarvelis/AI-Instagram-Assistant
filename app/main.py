import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import db
from app.admin import router as admin_router
from app.admin_ui import router as admin_ui_router
from app.config import get_settings
from app.telegram_webhook import router as telegram_router
from app.webhook import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the schema at startup and run the expiry sweep in the background.

    Schema creation is deliberately not wrapped in a try block: a database that
    cannot be opened at boot means a misconfigured volume, and crashing here
    surfaces that at deploy time rather than three weeks later.
    """
    db.init_schema()
    sweeper = asyncio.create_task(db.sweep_loop())
    try:
        yield
    finally:
        sweeper.cancel()


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    app = FastAPI(
        title="Tattoo Studio Instagram Assistant",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    app.include_router(admin_router)
    app.include_router(admin_ui_router)
    app.include_router(telegram_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()

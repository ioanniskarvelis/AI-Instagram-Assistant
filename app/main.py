import logging

from fastapi import FastAPI

from app.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    app = FastAPI(title="Tattoo Studio Instagram Assistant", version="1.0.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()

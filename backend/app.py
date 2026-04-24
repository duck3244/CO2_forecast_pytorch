"""FastAPI application entrypoint for CO2 forecast MVP."""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.config import backend_root
from api.routers import datasets, evaluations, models, predictions, training

STATIC_DIR = backend_root() / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="CO2 Forecast API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


app.include_router(datasets.router)
app.include_router(models.router)
app.include_router(predictions.router)
app.include_router(evaluations.router)
app.include_router(training.router)


# Static / SPA fallback — mounted AFTER API routers so /api/* wins.
# Only enabled if a production build exists in backend/static/.
if STATIC_DIR.exists():
    assets_dir = STATIC_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str, request: Request) -> FileResponse:
        # Let any /api/* call 404 cleanly rather than silently serving HTML.
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="not found")
        # Try an exact static file hit first (e.g. /vite.svg, /favicon.ico)
        candidate = STATIC_DIR / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        # Otherwise return the SPA shell; client-side router takes over.
        index_html = STATIC_DIR / "index.html"
        if not index_html.exists():
            raise HTTPException(status_code=404, detail="build not found")
        return FileResponse(index_html)

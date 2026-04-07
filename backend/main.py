"""
Entry point for the Math Photo Solver API.

Run locally:
    uvicorn backend.main:app --reload --port 8000

Then open http://localhost:8000 in a browser (serves frontend/index.html)
or call the API at http://localhost:8000/solve
"""

import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router

app = FastAPI(
    title="Math Photo Solver",
    description="Upload a photo of a math problem — get a step-by-step solution.",
    version="1.0.0",
)

# Allow requests from any origin (needed for mobile / Telegram WebView later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(router)

# Serve the frontend SPA
_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(os.path.join(_FRONTEND_DIR, "index.html"))

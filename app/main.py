"""
app/main.py
===========
Application entry point for the Multi-Agent AI Research System.

Responsibilities:
    - Creates and configures the FastAPI application instance
    - Registers CORS middleware for cross-origin API access
    - Mounts the API router (all endpoints defined in app/api/routes.py)
    - Mounts the Gradio web UI at /ui (http://localhost:8000/ui)
    - Ensures the output directory exists on startup
    - Provides the uvicorn entrypoint for local development

Usage:
    python -m app.main           # starts server on http://0.0.0.0:8000
    uvicorn app.main:app         # alternative direct uvicorn invocation

Interfaces:
    REST API:   http://localhost:8000/docs   (Swagger UI)
    Gradio UI:  http://localhost:8000/ui     (browser-based research interface)

No inputs or outputs — this module is the runtime bootstrap.
"""

import logging
import os
import uvicorn
import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load .env file before any module imports that read env vars (e.g. agents)
load_dotenv()

from app.config import OUTPUT_DIR, SERVER_HOST, SERVER_PORT, UI_URL, GROQ_API_KEY, TAVILY_API_KEY
from app.api.routes import router
from app.ui.gradio_app import build_gradio_app
from app.utils.logging_config import configure_logging
from app.utils.metrics import metrics

# Configure logging as early as possible
configure_logging()
_logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Factory function that builds and returns the configured FastAPI application.

    The application includes:
        - All REST API routes from app/api/routes.py
        - The Gradio browser UI mounted at /ui

    Returns:
        FastAPI: Fully configured application instance ready for serving.
    """
    app = FastAPI(
        title="Multi-Agent Research System",
        description="AI-powered research system using LangGraph and Claude",
        version="0.1.0",
    )

    # Allow all origins for MVP — restrict to specific domains in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register all API routes (/research, /approve, /status, /download, /jobs)
    app.include_router(router)

    # Mount the Gradio UI — accessible at http://localhost:8000/ui
    # gr.mount_gradio_app returns the FastAPI app with the Gradio routes added
    gradio_app = build_gradio_app()
    app = gr.mount_gradio_app(app, gradio_app, path="/ui")

    @app.get("/health")
    async def health():
        """Liveness probe — returns 200 when the server is up."""
        return {"status": "ok"}

    @app.get("/metrics")
    async def get_metrics():
        """Runtime metrics: request counts, averages, cost totals."""
        return metrics.get_all()

    @app.on_event("startup")
    async def startup():
        # ── Validate critical config ──────────────────────────────────────────
        missing = []
        if not GROQ_API_KEY:
            missing.append("GROQ_API_KEY")
        if not TAVILY_API_KEY:
            missing.append("TAVILY_API_KEY")
        if missing:
            _logger.error("STARTUP FAILED — missing required env vars: %s", ", ".join(missing))
            raise RuntimeError(f"Missing required config: {', '.join(missing)}")

        # ── Ensure output dir exists and is writable ──────────────────────────
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        try:
            test_file = os.path.join(OUTPUT_DIR, ".write_test")
            with open(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)
        except OSError as exc:
            _logger.error("OUTPUT_DIR '%s' is not writable: %s", OUTPUT_DIR, exc)
            raise

        # ── Log startup config (mask key values) ─────────────────────────────
        from app.config import (
            GROQ_MODEL, MAX_SUBAGENTS, MAX_ITERATIONS,
            MAX_COST_PER_QUERY, LOG_FORMAT, LOG_LEVEL,
        )
        _logger.info("=== Multi-Agent Research System starting ===")
        _logger.info("GROQ_API_KEY:      %s", "set" if GROQ_API_KEY else "MISSING")
        _logger.info("TAVILY_API_KEY:    %s", "set" if TAVILY_API_KEY else "MISSING")
        _logger.info("GROQ_MODEL:        %s", GROQ_MODEL)
        _logger.info("MAX_SUBAGENTS:     %s", MAX_SUBAGENTS)
        _logger.info("MAX_ITERATIONS:    %s", MAX_ITERATIONS)
        _logger.info("MAX_COST_PER_QUERY: $%.2f", MAX_COST_PER_QUERY)
        _logger.info("LOG_FORMAT:        %s", LOG_FORMAT)
        _logger.info("LOG_LEVEL:         %s", LOG_LEVEL)
        _logger.info("UI:                %s", UI_URL)
        _logger.info("API docs:          http://%s:%s/docs", SERVER_HOST, SERVER_PORT)

    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=SERVER_HOST, port=SERVER_PORT, reload=True)

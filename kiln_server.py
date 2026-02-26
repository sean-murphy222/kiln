"""Kiln unified backend server.

Mounts all four Kiln tool backends (Quarry, Forge, Foundry, Hearth)
under a single FastAPI application. Each tool's router is imported
lazily so that missing optional dependencies do not prevent the
server from starting -- the affected tool's endpoints simply return
503 and the unified health endpoint reports which tools loaded.

Usage::

    # Development (auto-reload)
    uvicorn kiln_server:app --reload --port 8420

    # Production
    uvicorn kiln_server:app --host 0.0.0.0 --port 8420

    # Or run directly
    python kiln_server.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("kiln")

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Kiln API",
    description=(
        "Unified backend for the Kiln pipeline: "
        "Quarry (document processing), Forge (curriculum builder), "
        "Foundry (training & evaluation), Hearth (interaction layer)."
    ),
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# CORS -- allow local Electron / Vite dev server origins
# ---------------------------------------------------------------------------

_ALLOWED_ORIGINS = [
    "http://localhost:5173",   # Vite dev server
    "http://localhost:8420",   # Self (for Swagger UI)
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8420",
    "app://.",                 # Electron production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Tool loading state -- tracks which tools mounted successfully
# ---------------------------------------------------------------------------

_tool_status: dict[str, dict[str, Any]] = {
    "quarry": {"loaded": False, "error": None},
    "forge": {"loaded": False, "error": None},
    "foundry": {"loaded": False, "error": None},
    "hearth": {"loaded": False, "error": None},
}


# ---------------------------------------------------------------------------
# Quarry (document processing)
# ---------------------------------------------------------------------------


def _mount_quarry() -> None:
    """Mount the Quarry/CHONK router at ``/api/``.

    Quarry endpoints keep their original ``/api/...`` paths because
    the router paths already contain the ``/api/`` prefix.
    """
    try:
        from chonk.server import router as quarry_router

        app.include_router(quarry_router, tags=["quarry"])
        _tool_status["quarry"]["loaded"] = True
        logger.info("Quarry router mounted successfully")
    except Exception as exc:
        _tool_status["quarry"]["error"] = str(exc)
        logger.warning("Quarry router failed to load: %s", exc)


# ---------------------------------------------------------------------------
# Forge (curriculum builder)
# ---------------------------------------------------------------------------


def _mount_forge() -> None:
    """Mount the Forge router at ``/api/forge/``.

    Initializes ForgeStorage with an on-disk SQLite database
    in the project ``data/forge/`` directory.
    """
    try:
        from forge.src.server import init_forge_storage, router as forge_router

        data_dir = Path("data/forge")
        data_dir.mkdir(parents=True, exist_ok=True)
        db_path = data_dir / "forge.db"
        init_forge_storage(db_path)

        app.include_router(forge_router, prefix="/api/forge", tags=["forge"])
        _tool_status["forge"]["loaded"] = True
        logger.info("Forge router mounted at /api/forge/")
    except Exception as exc:
        _tool_status["forge"]["error"] = str(exc)
        logger.warning("Forge router failed to load: %s", exc)


# ---------------------------------------------------------------------------
# Foundry (training & evaluation)
# ---------------------------------------------------------------------------


def _mount_foundry() -> None:
    """Mount the Foundry router at ``/api/foundry/``.

    Foundry services self-initialize lazily via module-level
    ``_get_*`` helpers, so no explicit setup call is needed.
    """
    try:
        from foundry.src.server import router as foundry_router

        app.include_router(
            foundry_router, prefix="/api/foundry", tags=["foundry"]
        )
        _tool_status["foundry"]["loaded"] = True
        logger.info("Foundry router mounted at /api/foundry/")
    except Exception as exc:
        _tool_status["foundry"]["error"] = str(exc)
        logger.warning("Foundry router failed to load: %s", exc)


# ---------------------------------------------------------------------------
# Hearth (interaction layer)
# ---------------------------------------------------------------------------


def _mount_hearth() -> None:
    """Mount the Hearth router at ``/api/hearth/``.

    Creates default HearthEngine, DocumentBrowser, and FeedbackManager
    instances and injects them via the module's ``configure()`` function.
    """
    try:
        from hearth.src.feedback import FeedbackManager
        from hearth.src.inference import (
            DocumentBrowser,
            HearthEngine,
            ModelManager,
        )
        from hearth.src.server import configure, router as hearth_router

        # Create default instances for MVP
        model_manager = ModelManager()

        # RAGPipeline requires a model and retrieval adapter.
        # For the unified server MVP we initialize the engine
        # with a stub RAGPipeline via optional import.
        rag_pipeline = _create_default_rag_pipeline()

        engine = HearthEngine(
            model_manager=model_manager,
            rag_pipeline=rag_pipeline,
        )
        browser = DocumentBrowser()
        feedback_manager = FeedbackManager()

        configure(
            engine=engine,
            browser=browser,
            feedback_manager=feedback_manager,
        )

        app.include_router(
            hearth_router, prefix="/api/hearth", tags=["hearth"]
        )
        _tool_status["hearth"]["loaded"] = True
        logger.info("Hearth router mounted at /api/hearth/")
    except Exception as exc:
        _tool_status["hearth"]["error"] = str(exc)
        logger.warning("Hearth router failed to load: %s", exc)


def _create_default_rag_pipeline() -> Any:
    """Create a default RAGPipeline for the Hearth engine.

    Uses MockInference and a no-op retrieval adapter so that the
    server can start without real model weights.

    Returns:
        A RAGPipeline configured with stub components.
    """
    from foundry.src.evaluation import MockInference
    from foundry.src.rag_integration import RAGPipeline

    class _StubRetrieval:
        """Retrieval adapter that returns no results.

        Satisfies the ``RetrievalAdapter`` protocol. Used as a
        placeholder until Quarry documents are indexed.
        """

        def retrieve(
            self,
            query: str,
            filters: dict[str, Any] | None = None,
        ) -> list[dict[str, Any]]:
            """Return an empty result set.

            Args:
                query: The search query (unused).
                filters: Optional metadata filters (unused).

            Returns:
                Empty list.
            """
            return []

    model = MockInference(default_response="I don't know.")
    retrieval = _StubRetrieval()
    return RAGPipeline(model=model, retrieval=retrieval)


# ---------------------------------------------------------------------------
# Unified health endpoint (registered BEFORE tool routers so that
# /api/health resolves to the unified version rather than Quarry's)
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def unified_health() -> dict[str, Any]:
    """Return health status for all Kiln tools.

    Reports which tool backends loaded successfully and which
    encountered import or initialization errors. A tool that
    failed to load will have ``loaded: false`` and an ``error``
    message explaining the failure.

    Returns:
        Dictionary with overall status and per-tool breakdown.
    """
    all_loaded = all(t["loaded"] for t in _tool_status.values())
    any_loaded = any(t["loaded"] for t in _tool_status.values())

    if all_loaded:
        status = "ok"
    elif any_loaded:
        status = "degraded"
    else:
        status = "error"

    return {
        "status": status,
        "version": "0.1.0",
        "tools": _tool_status,
    }


# ---------------------------------------------------------------------------
# Mount all tools
# ---------------------------------------------------------------------------

_mount_quarry()
_mount_forge()
_mount_foundry()
_mount_hearth()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_server(host: str = "127.0.0.1", port: int = 8420) -> None:
    """Start the Kiln unified server via uvicorn.

    Args:
        host: Bind address. Defaults to localhost.
        port: Port number. Defaults to 8420.
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    run_server()

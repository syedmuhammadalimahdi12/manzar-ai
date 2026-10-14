"""
MANZAR — Traffic Accident Reconstruction Backend
==================================================
Entry point:  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Then open:  http://localhost:8000/MANZAR.html

Routers:
  /session/*     — create, join, QR code, session state
  /ws/*          — WebSocket real-time events
  /session/*/upload          — photo upload
  /session/*/reconstruct     — trigger 3D reconstruction
  /session/*/chat            — "Ask MANZAR" evidence assistant
  /session/*/mesh            — serve 3D mesh
  /session/*/pointcloud      — serve point cloud
  /session/*/photos          — list photos
  /session/*/report          — structured report
  /tunnel/status             — ngrok tunnel health
  /uploads/*                 — static file mount for uploaded images
  /MANZAR.html               — frontend UI (served from static/)
"""

import asyncio
import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.routes import session, upload, reconstruction, chat, assets
from app.routes import ws as ws_router
from app.tunnel import start_tunnel, tunnel_health_loop, stop_tunnel, get_tunnel_info

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("manzar")

# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown hooks
# ---------------------------------------------------------------------------
USE_TUNNEL = os.getenv("USE_TUNNEL", "false").lower() == "true"
SERVER_PORT = int(os.getenv("PORT", "8000"))
STATIC_DIR = Path(__file__).parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    logger.info("MANZAR backend starting up...")
    llm_key = os.getenv("LLM_API_KEY", "")
    logger.info(f"LLM_API_KEY present: {bool(llm_key)} (length {len(llm_key)} chars)")
    logger.info(f"LLM_MODEL: {os.getenv('LLM_MODEL', 'not set')}")

    # Ensure uploads directory exists
    os.makedirs("uploads", exist_ok=True)

    # Ensure static directory exists
    os.makedirs(STATIC_DIR, exist_ok=True)

    logger.info(f"Frontend available at: http://localhost:{SERVER_PORT}/MANZAR.html")

    # Start ngrok tunnel if configured
    tunnel_task = None
    if USE_TUNNEL:
        url = start_tunnel(SERVER_PORT)
        if url:
            logger.info(f"Tunnel active: {url}")
            tunnel_task = asyncio.create_task(tunnel_health_loop(SERVER_PORT))
        else:
            logger.warning("Tunnel requested but failed to start")

    yield

    # --- Shutdown ---
    logger.info("MANZAR backend shutting down...")
    if USE_TUNNEL:
        stop_tunnel()
        if tunnel_task:
            tunnel_task.cancel()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MANZAR Backend",
    description="AI-powered traffic accident reconstruction — backend API + frontend",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — open for hackathon (all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(session.router, tags=["session"])
app.include_router(upload.router, tags=["upload"])
app.include_router(ws_router.router, tags=["websocket"])
app.include_router(reconstruction.router, tags=["reconstruction"])
app.include_router(chat.router, tags=["chat"])
app.include_router(assets.router, tags=["assets"])

# ---------------------------------------------------------------------------
# Static files — serve uploaded images
# ---------------------------------------------------------------------------
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/reconstruction", StaticFiles(directory="reconstruction"), name="reconstruction")


# ---------------------------------------------------------------------------
# Root & Frontend serving
# ---------------------------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "MANZAR backend is running", "version": "1.0.0", "frontend": "/MANZAR.html"}


@app.get("/tunnel/status")
def tunnel_status():
    return get_tunnel_info()


# Serve the frontend at /MANZAR.html
@app.get("/MANZAR.html")
def serve_frontend():
    response = FileResponse(STATIC_DIR / "MANZAR.html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# Serve static assets (CSS, JS, images if any)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

"""
MANZAR — ngrok Tunnel Manager  (NFR-9)
========================================
Manages an ngrok tunnel to expose the FastAPI backend during the live demo.
Features:
  - Auto-start tunnel on app startup (if USE_TUNNEL=true)
  - Health check every 10 minutes
  - Auto-reconnect if tunnel drops
  - GET /tunnel/status endpoint
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

logger = logging.getLogger("manzar.tunnel")

# ---------------------------------------------------------------------------
# Tunnel state
# ---------------------------------------------------------------------------

_tunnel_url: str | None = None
_tunnel_started_at: float | None = None
_tunnel_task: asyncio.Task | None = None
_ngrok_tunnel: Any = None


def get_tunnel_info() -> dict[str, Any]:
    """Return current tunnel status."""
    uptime = 0
    if _tunnel_started_at:
        uptime = round(time.time() - _tunnel_started_at, 0)

    return {
        "active": _tunnel_url is not None,
        "url": _tunnel_url,
        "uptime_seconds": uptime,
        "started_at": _tunnel_started_at,
    }


# ---------------------------------------------------------------------------
# Start tunnel
# ---------------------------------------------------------------------------

def start_tunnel(port: int = 8000) -> str | None:
    """Start an ngrok tunnel. Returns the public URL or None on failure."""
    global _tunnel_url, _tunnel_started_at, _ngrok_tunnel

    try:
        from pyngrok import ngrok

        # Set auth token if provided
        auth_token = os.getenv("NGROK_AUTH_TOKEN")
        if auth_token:
            ngrok.set_auth_token(auth_token)

        # Open tunnel
        tunnel = ngrok.connect(port, "http")
        _ngrok_tunnel = tunnel
        _tunnel_url = tunnel.public_url
        _tunnel_started_at = time.time()

        logger.info(f"ngrok tunnel started: {_tunnel_url}")
        print(f"\n{'='*60}")
        print(f"  MANZAR TUNNEL URL: {_tunnel_url}")
        print(f"  (Share this URL with the frontend)")
        print(f"{'='*60}\n")

        return _tunnel_url

    except ImportError:
        logger.warning("pyngrok not installed — run: pip install pyngrok")
        return None
    except Exception as e:
        logger.error(f"Failed to start ngrok tunnel: {e}")
        return None


# ---------------------------------------------------------------------------
# Health check loop (runs as background task)
# ---------------------------------------------------------------------------

async def tunnel_health_loop(port: int = 8000, interval: int = 600) -> None:
    """
    Periodically check if the tunnel is still alive.
    Auto-reconnect if it drops. Runs every `interval` seconds (default: 10 min).
    """
    global _tunnel_url, _tunnel_started_at

    while True:
        await asyncio.sleep(interval)

        if _tunnel_url is None:
            logger.warning("Tunnel is down — attempting reconnect")
            new_url = start_tunnel(port)
            if new_url:
                logger.info(f"Tunnel reconnected: {new_url}")
            else:
                logger.error("Tunnel reconnection failed")
        else:
            uptime = round(time.time() - (_tunnel_started_at or 0), 0)
            logger.info(f"Tunnel healthy: {_tunnel_url} (uptime: {uptime}s)")


# ---------------------------------------------------------------------------
# Stop tunnel
# ---------------------------------------------------------------------------

def stop_tunnel() -> None:
    """Close the ngrok tunnel."""
    global _tunnel_url, _tunnel_started_at, _ngrok_tunnel

    if _ngrok_tunnel:
        try:
            from pyngrok import ngrok
            ngrok.disconnect(_ngrok_tunnel.public_url)
            ngrok.kill()
        except Exception:
            pass

    _tunnel_url = None
    _tunnel_started_at = None
    _ngrok_tunnel = None
    logger.info("ngrok tunnel stopped")

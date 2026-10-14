"""
MANZAR — Shared Session Store
==============================
Single source of truth for session data.
All routes and services import `sessions` and `connection_manager` from here,
so every module sees the same live data.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket


# ---------------------------------------------------------------------------
# In-memory session store  (fine for hackathon — swap for Redis/DB in prod)
# ---------------------------------------------------------------------------
sessions: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------
class ConnectionManager:
    """Manages WebSocket connections grouped by session_id."""

    def __init__(self) -> None:
        # session_id -> list[WebSocket]
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.setdefault(session_id, []).append(websocket)

    def disconnect(self, session_id: str, websocket: WebSocket) -> None:
        conns = self._connections.get(session_id, [])
        if websocket in conns:
            conns.remove(websocket)

    async def send_event(
        self, session_id: str, event_type: str, data: dict | None = None
    ) -> None:
        """Push a named event to every client in a session."""
        payload = json.dumps({
            "event": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data or {},
        })
        dead: list[WebSocket] = []
        for ws in self._connections.get(session_id, []):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        # Clean up broken connections
        for ws in dead:
            self.disconnect(session_id, ws)


# Global singleton — import this everywhere
connection_manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Helper: create a new session
# ---------------------------------------------------------------------------
def create_session(
    session_id: str,
    location: str = "",
    gps_lat: float = 0.0,
    gps_lng: float = 0.0,
    officer_name: str = "",
    officer_badge: str = "",
    case_number: str = "",
    officer_notes: str = "",
) -> dict[str, Any]:
    session: dict[str, Any] = {
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "photos": [],
        "phones_joined": [],
        "coverage_status": "pending",
        "missing_angles": [],
        "status": "waiting_for_photos",
        "reconstruction": None,
        "detections": [],
        "cross_angle_results": {},
        "report": None,
        # Investigation metadata
        "location": location or "Unknown Location",
        "gps_lat": gps_lat,
        "gps_lng": gps_lng,
        "officer_name": officer_name or "Investigating Officer",
        "officer_badge": officer_badge or "N/A",
        "case_number": case_number or f"MANZAR-{datetime.now().strftime('%Y%m%d')}-{session_id[:8].upper()}",
        "officer_notes": officer_notes or "",  # Initial notes/observations from officer
        # Timeline of events
        "timeline": [
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "event": "session_created",
                "detail": f"Investigation session {session_id} initiated",
                "icon": "start",
            }
        ],
    }
    sessions[session_id] = session
    return session

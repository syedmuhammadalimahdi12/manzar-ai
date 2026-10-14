"""
MANZAR — Session routes
========================
Create sessions, generate QR codes, retrieve session state.
"""

import os
import re
import socket
import uuid

import qrcode
import io
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse

from app.store import sessions, create_session, connection_manager

router = APIRouter()

# Frontend base URL — set via env var, defaults to localhost for dev
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")


def _get_lan_ip():
    """Auto-detect the current LAN IP address."""
    try:
        # Connect to a public address to determine the outgoing interface IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(1)
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        try:
            # Fallback: read from ifconfig / ip addr
            import subprocess
            result = subprocess.run(["ifconfig"], capture_output=True, text=True)
            ips = re.findall(
                r"inet (192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)",
                result.stdout,
            )
            return ips[0] if ips else "127.0.0.1"
        except Exception:
            return "127.0.0.1"


def _get_qr_base_url():
    """Return the best URL for QR codes. Auto-corrects stale local IPs."""
    env_url = FRONTEND_URL.strip()
    # If env URL is localhost, use auto-detected LAN IP
    if "localhost" in env_url or "127.0.0.1" in env_url:
        return f"http://{_get_lan_ip()}:8000"
    # If env URL contains a local IP, check if it still matches current IP
    local_ip_match = re.search(r"https?://(192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)", env_url)
    if local_ip_match:
        current_ip = _get_lan_ip()
        if current_ip != "127.0.0.1" and local_ip_match.group(1) != current_ip:
            # Replace stale IP with current LAN IP, keep port if present
            port_match = re.search(r":(\d+)", env_url)
            port = port_match.group(1) if port_match else "8000"
            return f"http://{current_ip}:{port}"
    return env_url


@router.get("/sessions")
def list_sessions():
    """Return a list of all sessions for session history."""
    result = []
    for sid, s in sessions.items():
        photos = s.get("photos", [])
        covered = s.get("covered_angles", [])
        cross = s.get("cross_angle_results", [])
        consistency = "—"
        if isinstance(cross, dict) and "overall_score" in cross:
            consistency = f"{int(cross['overall_score'] * 100)}%"
        elif isinstance(cross, list) and cross:
            consistency = f"{len(cross)} items"
        result.append({
            "id": sid,
            "created_at": s.get("created_at", ""),
            "status": s.get("status", "unknown"),
            "photo_count": len(photos),
            "device_count": len(s.get("phones_joined", [])),
            "coverage": f"{len(covered)}/4" if covered else f"{s.get('coverage_status', 'pending')}",
            "consistency": consistency,
            "reconstruction_status": (s.get("reconstruction") or {}).get("status", "not started"),
            "location": s.get("location", ""),
            "case_number": s.get("case_number", ""),
            "officer_name": s.get("officer_name", ""),
            "detection_count": len(s.get("detections", [])),
        })
    # Sort by created_at descending
    result.sort(key=lambda x: x["created_at"], reverse=True)
    return {"sessions": result, "count": len(result)}

@router.post("/session/create")
def create_session_endpoint(
    location: str = "",
    gps_lat: float = 0.0,
    gps_lng: float = 0.0,
    officer_name: str = "",
    officer_badge: str = "",
    case_number: str = "",
    officer_notes: str = "",
):
    """Create a new capture session with investigation context."""
    session_id = str(uuid.uuid4())[:8]
    session = create_session(
        session_id,
        location=location,
        gps_lat=gps_lat,
        gps_lng=gps_lng,
        officer_name=officer_name,
        officer_badge=officer_badge,
        case_number=case_number,
        officer_notes=officer_notes,
    )
    return {
        "session_id": session_id,
        "status": session["status"],
        "case_number": session.get("case_number", ""),
        "location": session.get("location", ""),
    }


@router.get("/session/{session_id}")
def get_session(session_id: str):
    """Return the full session state — used by frontend + chatbot."""
    if session_id not in sessions:
        return JSONResponse(status_code=404, content={"error": "session not found"})
    return sessions[session_id]


@router.get("/session/{session_id}/qr")
def get_qr_code(session_id: str):
    """Generate a QR code that bystanders scan to join the session."""
    if session_id not in sessions:
        return JSONResponse(status_code=404, content={"error": "session not found"})

    join_url = f"{_get_qr_base_url()}/MANZAR.html?join={session_id}"

    qr = qrcode.make(join_url)
    buf = io.BytesIO()
    qr.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@router.post("/session/{session_id}/join")
async def join_session(session_id: str, request: Request):
    """Register a phone joining the session (called after QR scan)."""
    if session_id not in sessions:
        return JSONResponse(status_code=404, content={"error": "session not found"})

    # Accept JSON body or query params
    try:
        body = await request.json()
        device_id = body.get("device_id", body.get("device_name", "unknown"))
        device_name = body.get("device_name", device_id)
    except Exception:
        device_name = request.query_params.get("device_name", "unknown")
        device_id = device_name

    # Use device_id for dedup, device_name for display.
    phones = sessions[session_id]["phones_joined"]
    is_new_device = not any(
        p.get("device_id") == device_id for p in phones if isinstance(p, dict)
    )
    if is_new_device:
        from datetime import datetime, timezone
        if isinstance(phones, list) and (len(phones) == 0 or isinstance(phones[0], dict)):
            sessions[session_id]["phones_joined"] = phones
            sessions[session_id]["phones_joined"].append({"device_id": device_id, "device_name": device_name})
        else:
            # Migrate old string-based phones to the current payload shape.
            new_phones = [{"device_id": p, "device_name": p} for p in phones]
            new_phones.append({"device_id": device_id, "device_name": device_name})
            sessions[session_id]["phones_joined"] = new_phones
        sessions[session_id].setdefault("timeline", []).append({
            "time": datetime.now(timezone.utc).isoformat(),
            "event": "device_joined",
            "detail": f"Device '{device_name}' joined investigation ({len(sessions[session_id]['phones_joined'])} devices total)",
            "icon": "phone",
        })

        await connection_manager.send_event(session_id, "device_joined", {
            "device_id": device_id,
            "device_name": device_name,
            "total_devices": len(sessions[session_id]["phones_joined"]),
        })

    return {
        "status": "joined" if is_new_device else "already_joined",
        "session_id": session_id,
        "devices_count": len(sessions[session_id]["phones_joined"]),
    }

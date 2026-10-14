"""
MANZAR — WebSocket routes
==========================
Real-time event streaming for live session updates.
Frontend connects to /ws/{session_id} and receives events like:
  - device_joined
  - photo_received
  - coverage_update / coverage_met
  - reconstruction_started / reconstruction_progress / reconstruction_done
  - detection_done
  - chat_response
  - error
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.store import sessions, connection_manager

router = APIRouter()


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    # Validate session
    if session_id not in sessions:
        await websocket.close(code=4004, reason="session not found")
        return

    await connection_manager.connect(session_id, websocket)

    try:
        # Send current session state on connect
        session = sessions[session_id]
        await websocket.send_json({
            "event": "session_state",
            "data": {
                "status": session["status"],
                "photo_count": len(session["photos"]),
                "devices_count": len(session["phones_joined"]),
                "coverage_status": session["coverage_status"],
                "missing_angles": session.get("missing_angles", []),
            },
        })

        # Keep connection alive, listen for client messages (e.g. ping)
        while True:
            data = await websocket.receive_text()
            # Client can send "ping" or commands — respond with pong
            if data == "ping":
                await websocket.send_json({"event": "pong"})

    except WebSocketDisconnect:
        connection_manager.disconnect(session_id, websocket)
    except Exception:
        connection_manager.disconnect(session_id, websocket)

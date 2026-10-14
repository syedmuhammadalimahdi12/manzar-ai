"""
MANZAR — Reconstruction routes
================================
Endpoints to trigger and monitor 3D reconstruction.
"""

import asyncio

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.store import sessions, connection_manager
from app.services.pipeline import run_reconstruction

router = APIRouter()


@router.post("/session/{session_id}/reconstruct")
async def trigger_reconstruction(
    session_id: str,
    background_tasks: BackgroundTasks,
):
    """
    Kick off 3D reconstruction for a session.
    Returns immediately; progress is streamed via WebSocket.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="session not found")

    session = sessions[session_id]

    # Guard: only accident-relevant images may enter reconstruction.
    eligible_photos = [
        photo for photo in session.get("photos", [])
        if not photo.get("is_irrelevant", False)
    ]
    photo_count = len(eligible_photos)
    if photo_count < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "At least 2 relevant, overlapping photos are required to start "
                f"3D reconstruction. This session currently has {photo_count}."
            ),
        )

    # Guard: don't restart if already running or done
    if session["status"] in ("reconstructing", "reconstruction_done", "reconstruction_started"):
        return JSONResponse(
            status_code=409,
            content={
                "error": "reconstruction already in progress or completed",
                "status": session["status"],
            },
        )

    # Run pipeline in the background
    background_tasks.add_task(_run_pipeline_safe, session_id)

    return {
        "status": "reconstruction_queued",
        "session_id": session_id,
        "photo_count": len(session["photos"]),
    }


async def _run_pipeline_safe(session_id: str) -> None:
    """Wrapper that catches pipeline errors and updates session state."""
    try:
        await run_reconstruction(session_id)
    except Exception as e:
        # pipeline.py already handles error events, but belt-and-suspenders
        session = sessions.get(session_id)
        if session:
            session["status"] = "reconstruction_failed"
        await connection_manager.send_event(session_id, "error", {
            "stage": "reconstruction",
            "message": str(e),
        })


@router.get("/session/{session_id}/reconstruction/status")
def get_reconstruction_status(session_id: str):
    """Return the current reconstruction state for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="session not found")

    session = sessions[session_id]
    return {
        "status": session["status"],
        "reconstruction": session.get("reconstruction"),
    }


@router.post("/session/{session_id}/mesh-upload")
async def upload_kaggle_mesh(
    session_id: str,
    mesh: UploadFile | None = None,
    pointcloud: UploadFile | None = None,
    sparse_pointcloud: UploadFile | None = None,
):
    """
    Upload mesh/point cloud files generated externally (e.g., Kaggle notebook).
    Accepts .obj, .ply, .glb files and installs them into the session.
    """
    import tempfile
    import os

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="session not found")

    from app.services.kaggle_mesh import install_mesh_files

    # Save uploaded files to temp dir, then install
    tmp_dir = tempfile.mkdtemp(prefix="manzar_mesh_")
    mesh_path = pc_path = sparse_path = None

    try:
        if mesh:
            mesh_path = os.path.join(tmp_dir, mesh.filename)
            with open(mesh_path, "wb") as f:
                content = await mesh.read()
                f.write(content)

        if pointcloud:
            pc_path = os.path.join(tmp_dir, pointcloud.filename)
            with open(pc_path, "wb") as f:
                content = await pointcloud.read()
                f.write(content)

        if sparse_pointcloud:
            sparse_path = os.path.join(tmp_dir, sparse_pointcloud.filename)
            with open(sparse_path, "wb") as f:
                content = await sparse_pointcloud.read()
                f.write(content)

        result = install_mesh_files(
            session_id,
            mesh_path=mesh_path,
            pointcloud_path=pc_path,
            sparse_pointcloud_path=sparse_path,
        )

        # Notify frontend via WebSocket
        await connection_manager.send_event(session_id, "reconstruction_done", {
            "mesh_available": "mesh" in result.get("installed", {}),
            "pointcloud_available": "pointcloud" in result.get("installed", {}),
            "backend": "kaggle",
        })

        return result

    finally:
        # Clean up temp files
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

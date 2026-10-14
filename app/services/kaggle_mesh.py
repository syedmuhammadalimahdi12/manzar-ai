"""
MANZAR — Kaggle Mesh Bridge
==============================
Bridges the gap between Kaggle 3D reconstruction output and the local backend.

Workflow:
  1. User runs the Kaggle notebook (MANZAR_Fast_Sparse_Mesh.ipynb)
  2. Kaggle generates mesh (.obj / .ply) + point cloud (.ply)
  3. User downloads the output ZIP from Kaggle
  4. User calls POST /session/{id}/mesh-upload with the files
  5. Files are placed in the local reconstruction workspace
  6. Frontend can now display the 3D model

Alternatively, if the Kaggle notebook is configured with a callback URL,
it can POST files directly to the backend.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger("manzar.kaggle_mesh")


def install_mesh_files(
    session_id: str,
    mesh_path: str | None = None,
    pointcloud_path: str | None = None,
    sparse_pointcloud_path: str | None = None,
) -> dict[str, Any]:
    """
    Install externally generated mesh files into the session's reconstruction workspace.
    Returns updated reconstruction metadata.
    """
    from app.store import sessions

    session = sessions.get(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found")

    work = Path(os.getenv("RECON_WORK_DIR", "reconstruction")) / session_id
    mesh_dir = work / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    recon = session.get("reconstruction") or {}
    installed = {}

    if mesh_path and os.path.exists(mesh_path):
        ext = mesh_path.rsplit(".", 1)[-1].lower()
        dest = mesh_dir / f"mesh.{ext}"
        shutil.copy2(mesh_path, dest)
        recon["mesh_path"] = str(dest)
        installed["mesh"] = str(dest)
        logger.info(f"Mesh installed: {dest}")

    if pointcloud_path and os.path.exists(pointcloud_path):
        dest = mesh_dir / "pointcloud.ply"
        shutil.copy2(pointcloud_path, dest)
        recon["pointcloud_path"] = str(dest)
        installed["pointcloud"] = str(dest)
        logger.info(f"Point cloud installed: {dest}")

    if sparse_pointcloud_path and os.path.exists(sparse_pointcloud_path):
        dest = mesh_dir / "sparse_points.ply"
        shutil.copy2(sparse_pointcloud_path, dest)
        recon["sparse_pointcloud_path"] = str(dest)
        installed["sparse_pointcloud"] = str(dest)
        logger.info(f"Sparse point cloud installed: {dest}")

    recon["status"] = "completed"
    recon["backend"] = "kaggle"
    session["reconstruction"] = recon
    session["status"] = "reconstruction_done"

    logger.info(f"Kaggle mesh files installed for session {session_id}: {installed}")
    return {
        "session_id": session_id,
        "installed": installed,
        "reconstruction": recon,
    }

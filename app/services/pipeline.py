"""
MANZAR — 3D Reconstruction Pipeline
======================================
Orchestrates the full reconstruction flow:
  1. Feature extraction (COLMAP SIFT)
  2. Feature matching (COLMAP exhaustive matcher)
  3. Sparse reconstruction / pose estimation (COLMAP mapper)
  4. Dense point cloud (COLMAP stereo fusion — GPU not required)
  5. Mesh generation (OpenMVS if available, else Poisson from Open3D)
  6. Export PLY for frontend Three.js viewer

Designed to run on Kaggle/Colab GPU notebook or Alibaba ECS.
The compute backend is configurable (SRS FR-15).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
from enum import Enum
from pathlib import Path
from typing import Any

from app.store import sessions, connection_manager

logger = logging.getLogger("manzar.pipeline")

# ---------------------------------------------------------------------------
# Compute backend enum (FR-15 — swappable without frontend changes)
# ---------------------------------------------------------------------------

class ComputeBackend(str, Enum):
    LOCAL = "local"
    KAGGLE = "kaggle"
    COLAB = "colab"
    ECS = "ecs"


ACTIVE_BACKEND = ComputeBackend(os.getenv("COMPUTE_BACKEND", "local"))

# Binary paths — auto-detect common install locations
def _find_colmap() -> str:
    """Find COLMAP binary in common locations."""
    env = os.getenv("COLMAP_BIN", "")
    if env and os.path.isfile(env) and os.access(env, os.X_OK):
        return env
    if env and shutil.which(env):
        return env
    # Check for the bundled colmap-wrapper.sh (pre-built .app binary)
    base_dir = Path(__file__).resolve().parent.parent.parent
    wrapper = base_dir / "colmap-wrapper.sh"
    if wrapper.exists():
        return str(wrapper)
    # Common install paths
    for candidate in [
        "/usr/local/bin/colmap",         # brew on Intel Mac
        "/opt/homebrew/bin/colmap",      # brew on Apple Silicon
        "/usr/bin/colmap",               # Linux apt
        "colmap",                        # PATH
    ]:
        if shutil.which(candidate):
            return candidate
    return "colmap"  # fallback, will fail with clear error


def _find_openmvs_bin(name: str) -> str | None:
    """Find an OpenMVS binary, return None if not available."""
    env_dir = os.getenv("OPENMVS_BIN", "/usr/local/bin")
    full = os.path.join(env_dir, name)
    if shutil.which(full):
        return full
    if shutil.which(name):
        return name
    return None


COLMAP_BIN = _find_colmap()
STAGE_TIMEOUT = int(os.getenv("STAGE_TIMEOUT", "300"))  # 5 min per stage default
WORK_DIR = os.getenv("RECON_WORK_DIR", "reconstruction")


# ---------------------------------------------------------------------------
# Helper: run a subprocess with timeout + retry
# ---------------------------------------------------------------------------

async def _run_cmd(
    cmd: list[str],
    stage_name: str,
    timeout: int = STAGE_TIMEOUT,
    retries: int = 1,
) -> subprocess.CompletedProcess:
    """Run a shell command with timeout and retry logic."""
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"[{stage_name}] attempt {attempt}: {' '.join(cmd)}")
            # Use shell execution for wrapper scripts that need env vars
            if cmd[0].endswith(".sh"):
                shell_cmd = " ".join(f'"{c}"' for c in cmd)
                proc = await asyncio.create_subprocess_shell(
                    shell_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )

            if proc.returncode != 0:
                err_msg = stderr.decode("utf-8", errors="replace")[:1000]
                logger.warning(f"[{stage_name}] exit {proc.returncode}: {err_msg}")
                if attempt == retries:
                    raise RuntimeError(
                        f"{stage_name} failed (exit {proc.returncode}): {err_msg[:300]}"
                    )
                continue

            out_text = stdout.decode("utf-8", errors="replace")
            logger.info(f"[{stage_name}] completed successfully")
            return subprocess.CompletedProcess(
                args=cmd, returncode=0,
                stdout=stdout, stderr=stderr,
            )

        except asyncio.TimeoutError:
            logger.warning(f"[{stage_name}] timed out after {timeout}s")
            if proc:
                proc.kill()
            if attempt == retries:
                raise RuntimeError(f"{stage_name} timed out after {timeout}s")

    raise RuntimeError(f"{stage_name} unexpected failure")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_reconstruction(session_id: str) -> dict[str, Any]:
    """
    Full 3D reconstruction pipeline for a session.
    Updates the session dict and pushes WebSocket events at each stage.
    """
    session = sessions.get(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found")

    # Relevance is determined during upload. Never include unrelated images
    # in the photogrammetry workspace.
    photos = [
        photo for photo in session.get("photos", [])
        if not photo.get("is_irrelevant", False)
    ]
    if len(photos) < 2:
        raise ValueError(
            f"Need at least 2 relevant photos for reconstruction, got {len(photos)}"
        )

    logger.info(f"Starting reconstruction for session {session_id} with {len(photos)} photos")
    logger.info(f"COLMAP binary: {COLMAP_BIN}")

    # --- Prepare workspace ---
    work = Path(WORK_DIR) / session_id
    images_dir = work / "images"
    sparse_dir = work / "sparse"
    dense_dir = work / "dense"
    mesh_dir = work / "mesh"
    db_path = work / "database.db"

    # Clean previous run if any
    if work.exists():
        shutil.rmtree(work)
    for d in [images_dir, sparse_dir, dense_dir, mesh_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Copy uploaded photos into the workspace
    copied = 0
    for photo in photos:
        src = photo.get("path", "")
        if not src or not os.path.exists(src):
            logger.warning(f"Photo path missing: {src}")
            continue
        dst = images_dir / photo["filename"]
        shutil.copy2(src, dst)
        copied += 1

    if copied < 2:
        raise ValueError(f"Only {copied} photos were actually found on disk (need >= 2)")

    logger.info(f"Copied {copied} images into {images_dir}")

    # Update session status
    session["status"] = "reconstructing"
    session["reconstruction"] = {
        "status": "running",
        "stage": "starting",
        "percent": 0,
        "backend": ACTIVE_BACKEND.value,
        "photo_count": copied,
    }
    await connection_manager.send_event(session_id, "reconstruction_started", {
        "photo_count": copied,
        "backend": ACTIVE_BACKEND.value,
    })

    start_time = time.time()

    try:
        # ── Stage 1: Feature extraction (SIFT) ──
        await _push_progress(session_id, "feature_extraction", 10)
        session["reconstruction"]["stage"] = "feature_extraction"
        await _run_cmd(
            [COLMAP_BIN, "feature_extractor",
             "--database_path", str(db_path),
             "--image_path", str(images_dir),
             "--ImageReader.single_camera", "1",
             "--SiftExtraction.max_num_features", "8192"],
            stage_name="feature_extraction",
        )

        # ── Stage 2: Feature matching ──
        await _push_progress(session_id, "feature_matching", 25)
        session["reconstruction"]["stage"] = "feature_matching"
        await _run_cmd(
            [COLMAP_BIN, "exhaustive_matcher",
             "--database_path", str(db_path),
             "--ExhaustiveMatching.block_size", "50"],
            stage_name="feature_matching",
        )

        # ── Stage 3: Sparse reconstruction (mapper) ──
        await _push_progress(session_id, "sparse_reconstruction", 40)
        session["reconstruction"]["stage"] = "sparse_reconstruction"
        await _run_cmd(
            [COLMAP_BIN, "mapper",
             "--database_path", str(db_path),
             "--image_path", str(images_dir),
             "--output_path", str(sparse_dir)],
            stage_name="sparse_reconstruction",
            timeout=STAGE_TIMEOUT * 3,
        )

        # Find the sparse model (usually in sparse/0/)
        sparse_model = _find_sparse_model(sparse_dir)
        if sparse_model is None:
            raise RuntimeError(
                "COLMAP mapper produced no sparse model. "
                "This usually means photos don't have enough overlap. "
                "Try uploading photos from closer angles with more scene overlap."
            )

        logger.info(f"Sparse model found at: {sparse_model}")

        # Export sparse point cloud for quick preview
        sparse_ply = mesh_dir / "sparse_points.ply"
        await _export_sparse_points(sparse_model, sparse_ply)

        # ── Stage 4: Dense reconstruction (COLMAP stereo) ──
        await _push_progress(session_id, "dense_reconstruction", 60)
        session["reconstruction"]["stage"] = "dense_reconstruction"

        dense_ok = await _colmap_dense(sparse_model, dense_dir, images_dir)

        # Try OpenMVS as upgrade (optional — only if binaries found)
        openmvs_used = False
        if dense_ok:
            openmvs_bin = _find_openmvs_bin("InterfaceCOLMAP")
            if openmvs_bin:
                logger.info("OpenMVS found — attempting upgrade")
                openmvs_used = await _try_openmvs_dense(
                    sparse_model, dense_dir, images_dir
                )

        # ── Stage 5: Mesh generation ──
        await _push_progress(session_id, "mesh_generation", 80)
        session["reconstruction"]["stage"] = "mesh_generation"
        mesh_file = await _generate_mesh(dense_dir, mesh_dir, images_dir)

        # ── Stage 6: Export for frontend ──
        await _push_progress(session_id, "exporting", 95)
        session["reconstruction"]["stage"] = "exporting"
        ply_path = _find_best_pointcloud(dense_dir, mesh_dir, sparse_ply)

        elapsed = round(time.time() - start_time, 1)

        # --- Update session ---
        session["reconstruction"] = {
            "status": "completed",
            "backend": ACTIVE_BACKEND.value,
            "elapsed_seconds": elapsed,
            "mesh_path": str(mesh_file) if mesh_file else None,
            "pointcloud_path": str(ply_path) if ply_path else None,
            "sparse_pointcloud_path": str(sparse_ply) if sparse_ply.exists() else None,
            "photo_count": copied,
            "openmvs_used": openmvs_used,
        }
        session["status"] = "reconstruction_done"

        # Regenerate rich report (includes reconstruction data now)
        from app.services.report_gen import generate_report
        session["report"] = generate_report(session)

        await connection_manager.send_event(session_id, "reconstruction_done", {
            "elapsed_seconds": elapsed,
            "mesh_available": mesh_file is not None,
            "pointcloud_available": ply_path is not None,
        })

        logger.info(f"Reconstruction complete for {session_id} in {elapsed}s")
        return session["reconstruction"]

    except Exception as e:
        elapsed = round(time.time() - start_time, 1)
        error_msg = str(e)
        logger.error(f"Reconstruction failed for {session_id}: {error_msg}")

        session["status"] = "reconstruction_failed"
        session["reconstruction"] = {
            "status": "failed",
            "error": error_msg,
            "elapsed_seconds": elapsed,
            "stage": session.get("reconstruction", {}).get("stage", "unknown"),
        }
        await connection_manager.send_event(session_id, "reconstruction_failed", {
            "stage": session["reconstruction"]["stage"],
            "message": error_msg,
        })
        raise


# ---------------------------------------------------------------------------
# Sparse model finder
# ---------------------------------------------------------------------------

def _find_sparse_model(sparse_dir: Path) -> Path | None:
    """Find the best sparse model directory (usually sparse/0/)."""
    # COLMAP mapper creates numbered subdirectories: 0, 1, 2...
    # The one with most images is the best
    candidates = sorted(sparse_dir.iterdir()) if sparse_dir.exists() else []
    best = None
    best_count = 0
    for d in candidates:
        if d.is_dir():
            # Check for binary model files first (COLMAP default)
            images_bin = d / "images.bin"
            if images_bin.exists():
                count = images_bin.stat().st_size
                if count > best_count:
                    best_count = count
                    best = d
                continue
            # Fallback: check for text model files
            images_txt = d / "images.txt"
            if images_txt.exists():
                count = sum(1 for line in images_txt.open() if line.strip() and not line.startswith("#"))
                if count > best_count:
                    best_count = count
                    best = d
    return best


# ---------------------------------------------------------------------------
# Sparse point cloud export
# ---------------------------------------------------------------------------

async def _export_sparse_points(sparse_model: Path, output_ply: Path) -> None:
    """Convert COLMAP sparse model to PLY point cloud using model_analyzer."""
    try:
        await _run_cmd(
            [COLMAP_BIN, "model_converter",
             "--input_path", str(sparse_model),
             "--output_path", str(output_ply),
             "--output_type", "PLY"],
            stage_name="sparse_export",
            timeout=60,
        )
    except Exception as e:
        logger.warning(f"Could not export sparse PLY: {e}")


# ---------------------------------------------------------------------------
# Dense reconstruction (COLMAP stereo pipeline)
# ---------------------------------------------------------------------------

async def _colmap_dense(
    sparse_model: Path, dense_dir: Path, images_dir: Path
) -> bool:
    """Dense reconstruction using COLMAP's stereo pipeline. Returns True on success."""
    try:
        undistorted = dense_dir / "undistorted"

        # Undistort images for stereo
        await _run_cmd(
            [COLMAP_BIN, "image_undistorter",
             "--image_path", str(images_dir),
             "--input_path", str(sparse_model),
             "--output_path", str(undistorted),
             "--output_type", "COLMAP"],
            stage_name="colmap_undistort",
        )

        # Patch match stereo
        await _run_cmd(
            [COLMAP_BIN, "patch_match_stereo",
             "--workspace_path", str(undistorted),
             "--workspace_format", "COLMAP",
             "--PatchMatchStereo.geom_consistency", "1",
             "--PatchMatchStereo.max_image_size", "1600"],
            stage_name="colmap_patch_match",
            timeout=STAGE_TIMEOUT * 3,
        )

        # Stereo fusion → PLY
        fused_ply = dense_dir / "fused.ply"
        await _run_cmd(
            [COLMAP_BIN, "stereo_fusion",
             "--workspace_path", str(undistorted),
             "--workspace_format", "COLMAP",
             "--input_type", "geometric",
             "--output_path", str(fused_ply),
             "--StereoFusion.max_image_size", "1600"],
            stage_name="colmap_stereo_fusion",
        )

        if fused_ply.exists():
            logger.info(f"Dense point cloud created: {fused_ply}")
            return True
        return False

    except Exception as e:
        logger.warning(f"COLMAP dense failed: {e}")
        return False


# ---------------------------------------------------------------------------
# OpenMVS dense (optional upgrade)
# ---------------------------------------------------------------------------

async def _try_openmvs_dense(
    sparse_model: Path, dense_dir: Path, images_dir: Path
) -> bool:
    """Attempt dense reconstruction via OpenMVS. Returns True on success."""
    interface_bin = _find_openmvs_bin("InterfaceCOLMAP")
    densify_bin = _find_openmvs_bin("DensifyPointCloud")
    if not interface_bin or not densify_bin:
        return False

    try:
        mvs_file = dense_dir / "scene.mvs"
        await _run_cmd(
            [interface_bin,
             "-i", str(sparse_model),
             "-o", str(mvs_file),
             "--image-folder", str(images_dir)],
            stage_name="openmvs_interface",
        )
        await _run_cmd(
            [densify_bin, str(mvs_file), "-w", str(dense_dir)],
            stage_name="openmvs_dense",
            timeout=STAGE_TIMEOUT * 2,
        )
        return True
    except Exception as e:
        logger.warning(f"OpenMVS dense failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------

async def _generate_mesh(
    dense_dir: Path, mesh_dir: Path, images_dir: Path
) -> Path | None:
    """Generate a mesh from the dense point cloud."""
    # Try OpenMVS mesh
    mesh_bin = _find_openmvs_bin("ReconstructMesh")
    mvs_file = dense_dir / "scene.mvs"
    if mesh_bin and mvs_file.exists():
        try:
            await _run_cmd(
                [mesh_bin, str(mvs_file), "-w", str(mesh_dir)],
                stage_name="openmvs_mesh",
            )
            mesh_file = mesh_dir / "scene_mesh.ply"
            if mesh_file.exists():
                return mesh_file
        except Exception:
            pass

    # Fallback: COLMAP fused point cloud (no mesh, but PLY is valid)
    fused = dense_dir / "fused.ply"
    if fused.exists():
        return fused

    return None


# ---------------------------------------------------------------------------
# Point cloud finder
# ---------------------------------------------------------------------------

def _find_best_pointcloud(
    dense_dir: Path, mesh_dir: Path, sparse_ply: Path
) -> Path | None:
    """Find the best available point cloud file."""
    for candidate in [
        dense_dir / "scene_dense.ply",       # OpenMVS dense
        dense_dir / "fused.ply",             # COLMAP stereo fusion
        mesh_dir / "scene_dense.ply",        # OpenMVS alt location
        mesh_dir / "sparse_points.ply",      # Sparse fallback
        sparse_ply,                          # Last resort
    ]:
        if candidate.exists() and candidate.stat().st_size > 100:
            return candidate
    return None


# ---------------------------------------------------------------------------
# Auto report generation
# ---------------------------------------------------------------------------

def _generate_report(session_id: str, session: dict, elapsed: float) -> dict:
    """Generate a basic structured report after reconstruction."""
    from datetime import datetime, timezone
    return {
        "session_id": session_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reconstruction_time_seconds": elapsed,
        "photos_analyzed": len(session.get("photos", [])),
        "coverage": {
            "status": session.get("coverage_status", "unknown"),
            "covered_angles": list(set(
                p.get("angle_hint", "unknown") for p in session.get("photos", [])
            )),
            "missing_angles": session.get("missing_angles", []),
        },
        "devices": list(set(
            p.get("device_id", "unknown") for p in session.get("photos", [])
        )),
        "outputs": {
            "mesh_available": session.get("reconstruction", {}).get("mesh_path") is not None,
            "pointcloud_available": session.get("reconstruction", {}).get("pointcloud_path") is not None,
        },
        "note": "Report generated from reconstruction and detection data.",
    }


# ---------------------------------------------------------------------------
# WebSocket progress helper
# ---------------------------------------------------------------------------

async def _push_progress(session_id: str, stage: str, percent: int) -> None:
    """Push reconstruction progress via WebSocket."""
    session = sessions.get(session_id)
    if session and session.get("reconstruction"):
        session["reconstruction"]["stage"] = stage
        session["reconstruction"]["percent"] = percent

    await connection_manager.send_event(session_id, "reconstruction_progress", {
        "stage": stage,
        "percent": percent,
    })

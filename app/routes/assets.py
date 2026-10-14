"""
MANZAR — Asset serving routes
===============================
Serve 3D meshes, point clouds, and annotated images to the frontend.
"""

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.store import sessions

router = APIRouter()


@router.get("/session/{session_id}/mesh")
def get_mesh(session_id: str):
    """Serve the reconstructed 3D mesh file (.ply or .obj)."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="session not found")

    recon = sessions[session_id].get("reconstruction") or {}
    mesh_path = recon.get("mesh_path")

    if not mesh_path or not os.path.exists(mesh_path):
        raise HTTPException(status_code=404, detail="mesh not available yet")

    # Determine media type
    ext = mesh_path.rsplit(".", 1)[-1].lower()
    media_types = {
        "ply": "application/octet-stream",
        "obj": "text/plain",
        "glb": "model/gltf-binary",
        "gltf": "model/gltf+json",
    }

    return FileResponse(
        path=mesh_path,
        media_type=media_types.get(ext, "application/octet-stream"),
        filename=f"manzar_{session_id}_mesh.{ext}",
    )


@router.get("/session/{session_id}/pointcloud")
def get_pointcloud(session_id: str):
    """Serve the dense point cloud as .ply for the Three.js viewer."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="session not found")

    recon = sessions[session_id].get("reconstruction") or {}
    pc_path = recon.get("pointcloud_path")

    if not pc_path or not os.path.exists(pc_path):
        raise HTTPException(status_code=404, detail="point cloud not available yet")

    return FileResponse(
        path=pc_path,
        media_type="application/octet-stream",
        filename=f"manzar_{session_id}_pointcloud.ply",
    )


@router.get("/session/{session_id}/photos")
def list_photos(session_id: str):
    """List all photos in a session with metadata."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="session not found")

    photos = sessions[session_id].get("photos", [])
    return {
        "session_id": session_id,
        "photo_count": len(photos),
        "photos": [
            {
                "filename": p["filename"],
                "angle_hint": p.get("angle_hint"),
                "device_id": p.get("device_id"),
                "url": f"/uploads/{session_id}/{p['filename']}",
            }
            for p in photos
        ],
    }


@router.get("/session/{session_id}/report")
def get_report(session_id: str):
    """Return the structured incident report for this session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="session not found")

    report = sessions[session_id].get("report")
    if not report:
        # Auto-generate report from current session data
        from app.services.report_gen import generate_report
        from app.services.cross_angle import run_cross_angle_verification
        session = sessions[session_id]
        # Ensure cross-angle is up to date
        if not session.get("cross_angle_results"):
            session["cross_angle_results"] = run_cross_angle_verification(session.get("photos", []))
        report = generate_report(session)
        session["report"] = report

    return {
        "session_id": session_id,
        "report": report,
    }


# --- Detection endpoint ---
@router.get("/session/{session_id}/detections")
def get_detections(session_id: str):
    """Return all detection results for a session."""
    from app.store import sessions
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    
    from app.services.detection import summarize_detections
    all_dets = {}
    for photo in session.get("photos", []):
        fn = photo.get("filename", "")
        dets = photo.get("detections", [])
        if dets:
            all_dets[fn] = dets
    
    return {
        "session_id": session_id,
        "detections_by_photo": all_dets,
        "summary": summarize_detections(all_dets),
    }


@router.post("/session/{session_id}/detect")
def run_detection(session_id: str):
    """Re-run YOLOv8 detection on all session photos."""
    from app.store import sessions
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    
    from app.services.detection import detect_all_photos, summarize_detections
    
    photos = session.get("photos", [])
    all_dets = detect_all_photos(photos)
    
    # Update photo records
    for photo in photos:
        fn = photo.get("filename", "")
        if fn in all_dets:
            photo["detections"] = all_dets[fn]
            summary = {}
            for d in all_dets[fn]:
                summary[d["class"]] = summary.get(d["class"], 0) + 1
            photo["detection_summary"] = summary
    
    return {
        "session_id": session_id,
        "detections_by_photo": all_dets,
        "summary": summarize_detections(all_dets),
    }


# --- Manual analysis trigger ---
@router.post("/session/{session_id}/analyze")
def run_analysis(session_id: str):
    """Run cross-angle verification and generate report on demand."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")

    from app.services.cross_angle import run_cross_angle_verification
    from app.services.report_gen import generate_report

    # Run cross-angle
    cross_angle = run_cross_angle_verification(session.get("photos", []))
    session["cross_angle_results"] = cross_angle

    # Generate report
    report = generate_report(session)
    session["report"] = report

    return {
        "session_id": session_id,
        "cross_angle": cross_angle,
        "report": report,
    }


# --- PDF report export ---
@router.get("/session/{session_id}/report/pdf")
def export_report_pdf(session_id: str):
    """Generate and download the incident report as a PDF."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="session not found")

    from app.services.report_pdf import generate_pdf
    from io import BytesIO
    from fastapi.responses import StreamingResponse

    session = sessions[session_id]
    report = session.get("report")
    if not report:
        from app.services.report_gen import generate_report
        from app.services.cross_angle import run_cross_angle_verification
        if not session.get("cross_angle_results"):
            session["cross_angle_results"] = run_cross_angle_verification(session.get("photos", []))
        report = generate_report(session)
        session["report"] = report

    pdf_bytes = generate_pdf(report, session)
    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=manzar_report_{session_id}.pdf"},
    )

"""
MANZAR — Upload routes
=======================
Handle photo/video uploads from multiple phones.
Each upload is linked back to the session and triggers a coverage check.

Hybrid detection: YOLO handles vehicle_damage; Gemini fills in missing classes
(skid_mark, debris_field, point_of_impact) with "ai_inferred" source marking.
"""

import os
import uuid
import logging
import asyncio
from datetime import datetime, timezone

logger = logging.getLogger("manzar.upload")

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from app.store import sessions, connection_manager
from app.services.coverage_check import check_coverage
from app.services.gemini_detect import gemini_analyze_photo
from app.services.cross_angle import run_cross_angle_verification
from app.services.report_gen import generate_report

router = APIRouter()
UPLOAD_DIR = "uploads"


@router.post("/session/{session_id}/upload")
async def upload_photo(
    session_id: str,
    background_tasks: BackgroundTasks,
    angle_hint: str = Form(None),
    device_id: str = Form("unknown"),
    gps_lat: float = Form(0.0),
    gps_lng: float = Form(0.0),
    file: UploadFile = File(...),
):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="session not found")

    session = sessions[session_id]
    session_folder = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_folder, exist_ok=True)

    safe_name = file.filename or "unknown.jpg"
    filename = f"{uuid.uuid4().hex[:8]}_{safe_name}"
    filepath = os.path.join(session_folder, filename)

    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # --- Compute photo hash for tamper detection ---
    import hashlib
    with open(filepath, "rb") as f:
        photo_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    # --- Create photo record immediately (analysis pending) ---
    upload_time = datetime.now(timezone.utc).isoformat()
    photo_record = {
        "filename": filename,
        "path": filepath,
        "angle_hint": angle_hint if angle_hint and angle_hint != "auto" else "unknown",
        "detected_angle": "unknown",
        "device_id": device_id,
        "detections": [],
        "detection_summary": {},
        "yolo_detections": 0,
        "ai_detections": 0,
        "photo_hash": photo_hash,
        "uploaded_at": upload_time,
        "gps_lat": gps_lat,
        "gps_lng": gps_lng,
        "photo_number": len(session["photos"]) + 1,
        "is_irrelevant": False,
        "relevance_reason": "",
        "analysis_pending": True,
    }
    session["photos"].append(photo_record)

    # --- Add timeline event ---
    session.setdefault("timeline", []).append({
        "time": upload_time,
        "event": "photo_uploaded",
        "detail": f"Photo from {angle_hint or 'unknown'} angle by {device_id} — analysis pending",
        "icon": "camera",
        "photo_number": photo_record["photo_number"],
    })

    # --- Push WebSocket event (immediate) ---
    await connection_manager.send_event(session_id, "photo_uploaded", {
        "filename": filename,
        "angle_hint": photo_record["angle_hint"],
        "detected_angle": "unknown",
        "device_id": device_id,
        "total_photos": len(session["photos"]),
        "detections": 0,
        "yolo_detections": 0,
        "ai_detections": 0,
        "detection_summary": {},
        "ai_classes": [],
        "is_irrelevant": False,
        "relevance_reason": "",
        "analysis_pending": True,
    })

    # --- Trigger background analysis (only if not already analyzed) ---
    background_tasks.add_task(_analyze_photo_background, session_id, filename)

    return {
        "status": "uploaded",
        "filename": filename,
        "angle_hint": photo_record["angle_hint"],
        "detected_angle": "unknown",
        "photo_count": len(session["photos"]),
        "coverage": {"sufficient": False, "missing_angles": ["front", "left", "rear", "right"], "covered_angles": [], "photo_count": len(session["photos"]), "untagged_photos": len(session["photos"]), "angle_breakdown": {}, "recommendation": "Click 'Start Investigation' to analyze photos"},
        "detections": [],
        "detection_summary": {},
        "yolo_detections": 0,
        "ai_detections": 0,
        "ai_classes": [],
        "is_irrelevant": False,
        "relevance_reason": "",
        "analysis_pending": True,
    }


async def _analyze_photo_background(session_id: str, filename: str):
    """Run Gemini analysis in background and update session."""
    if session_id not in sessions:
        return
    
    session = sessions[session_id]
    photo = next((p for p in session["photos"] if p["filename"] == filename), None)
    if not photo:
        return
    
    # Skip if already analyzed
    if not photo.get("analysis_pending", True):
        logger.info(f"Skipping analysis for {filename} - already completed")
        return
    
    filepath = photo["path"]
    logger.info(f"Starting background analysis for {filename}")
    
    try:
        # --- Run Gemini analysis ---
        analysis = await gemini_analyze_photo(filepath, conf_threshold=0.3)
        relevance = {"relevant": analysis["relevant"], "reason": analysis["reason"], "angle": analysis.get("angle", "unknown")}
        gemini_detections = analysis["detections"]
        
        # --- Update photo record ---
        photo["detections"] = gemini_detections
        photo["detection_summary"] = {}
        for d in gemini_detections:
            cls = d["class"]
            photo["detection_summary"][cls] = photo["detection_summary"].get(cls, 0) + 1
        photo["ai_detections"] = len(gemini_detections)
        photo["detected_angle"] = relevance.get("angle", "unknown")
        if photo["angle_hint"] == "unknown" and relevance.get("angle") and relevance["angle"] != "unknown":
            photo["angle_hint"] = relevance["angle"]
        photo["is_irrelevant"] = not relevance["relevant"]
        photo["relevance_reason"] = relevance.get("reason", "")
        photo["analysis_pending"] = False
        
        # --- Update timeline ---
        det_count = len(gemini_detections)
        det_classes = list(set(d["class"] for d in gemini_detections))
        detail = f"Photo from {photo['angle_hint'] or 'unknown'} angle by {photo['device_id']}"
        if det_count > 0:
            detail += f" — {det_count} AI evidence detected"
        elif relevance["relevant"]:
            detail += " — no evidence visible"
        else:
            detail += f" — rejected: {relevance.get('reason', 'not accident related')}"
        
        # Replace the pending timeline event
        for t in session.get("timeline", []):
            if t.get("event") == "photo_uploaded" and t.get("detail", "").endswith("analysis pending"):
                t["detail"] = detail
                break
        
        # --- Aggregate detections into session-level list ---
        session["detections"] = []
        for p in session["photos"]:
            if p.get("is_irrelevant"):
                continue
            for det in p.get("detections", []):
                session["detections"].append({
                    **det,
                    "source_photo": p["filename"],
                    "device_id": p.get("device_id", "unknown"),
                    "angle_hint": p.get("angle_hint", ""),
                })
        
        # --- Run cross-angle verification ---
        cross_angle = run_cross_angle_verification(session["photos"])
        session["cross_angle_results"] = cross_angle
        
        # --- Run coverage check ---
        coverage = check_coverage(session["photos"])
        session["coverage_status"] = "sufficient" if coverage["sufficient"] else "insufficient"
        session["missing_angles"] = coverage.get("missing_angles", [])
        session["covered_angles"] = coverage.get("covered_angles", [])
        
        # --- Auto-generate report if we have detections ---
        if session.get("detections") and len(session["detections"]) > 0:
            session["report"] = generate_report(session)
        
        # --- Push WebSocket event ---
        await connection_manager.send_event(session_id, "photo_analysis_complete", {
            "filename": filename,
            "angle_hint": photo["angle_hint"],
            "detected_angle": photo["detected_angle"],
            "detections": det_count,
            "ai_classes": det_classes,
            "is_irrelevant": photo["is_irrelevant"],
            "coverage": coverage,
        })
        
        logger.info(f"Background analysis complete for {filename}: {det_count} detections")
        
    except Exception as e:
        logger.error(f"Background analysis failed for {filename}: {e}")
        photo["analysis_pending"] = False
        photo["relevance_reason"] = f"Analysis failed: {str(e)}"


@router.post("/session/{session_id}/analyze-all")
async def analyze_all_photos(session_id: str):
    """Run Gemini analysis on ALL pending photos in PARALLEL, then return results."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="session not found")
    
    session = sessions[session_id]
    pending_photos = [p for p in session.get("photos", []) if p.get("analysis_pending", False)]
    
    if not pending_photos:
        return {"status": "no_pending_photos", "analyzed": 0}
    
    logger.info(f"Starting PARALLEL analysis for {len(pending_photos)} photos in session {session_id}")
    
    async def analyze_single_photo(photo):
        """Analyze a single photo and update its record."""
        filepath = photo["path"]
        try:
            analysis = await gemini_analyze_photo(filepath, conf_threshold=0.3)
            relevance = {"relevant": analysis["relevant"], "reason": analysis["reason"], "angle": analysis.get("angle", "unknown")}
            gemini_detections = analysis["detections"]
            
            # Update photo record
            photo["detections"] = gemini_detections
            photo["detection_summary"] = {}
            for d in gemini_detections:
                cls = d["class"]
                photo["detection_summary"][cls] = photo["detection_summary"].get(cls, 0) + 1
            photo["ai_detections"] = len(gemini_detections)
            photo["detected_angle"] = relevance.get("angle", "unknown")
            if photo["angle_hint"] == "unknown" and relevance.get("angle") and relevance["angle"] != "unknown":
                photo["angle_hint"] = relevance["angle"]
            photo["is_irrelevant"] = not relevance["relevant"]
            photo["relevance_reason"] = relevance.get("reason", "")
            photo["analysis_pending"] = False
            
            # Update timeline
            det_count = len(gemini_detections)
            detail = f"Photo from {photo['angle_hint'] or 'unknown'} angle by {photo['device_id']}"
            if det_count > 0:
                detail += f" — {det_count} AI evidence detected"
            elif relevance["relevant"]:
                detail += " — no evidence visible"
            else:
                detail += f" — rejected: {relevance.get('reason', 'not accident related')}"
            
            for t in session.get("timeline", []):
                if t.get("event") == "photo_uploaded" and t.get("detail", "").endswith("analysis pending"):
                    t["detail"] = detail
                    break
            
            logger.info(f"Analyzed {photo['filename']}: {det_count} detections")
            return {"filename": photo["filename"], "detections": det_count, "success": True}
        except Exception as e:
            logger.error(f"Analysis failed for {photo['filename']}: {e}")
            photo["analysis_pending"] = False
            photo["relevance_reason"] = f"Analysis failed: {str(e)}"
            return {"filename": photo["filename"], "detections": 0, "success": False}
    
    # Run all analyses in PARALLEL
    results = await asyncio.gather(*[analyze_single_photo(photo) for photo in pending_photos])
    results = list(results)
    
    # Aggregate detections
    session["detections"] = []
    for p in session["photos"]:
        if p.get("is_irrelevant"):
            continue
        for det in p.get("detections", []):
            session["detections"].append({
                **det,
                "source_photo": p["filename"],
                "device_id": p.get("device_id", "unknown"),
                "angle_hint": p.get("angle_hint", ""),
            })
    
    # Run cross-angle verification
    cross_angle = run_cross_angle_verification(session["photos"])
    session["cross_angle_results"] = cross_angle
    
    # Run coverage check
    coverage = check_coverage(session["photos"])
    session["coverage_status"] = "sufficient" if coverage["sufficient"] else "insufficient"
    session["missing_angles"] = coverage.get("missing_angles", [])
    session["covered_angles"] = coverage.get("covered_angles", [])
    
    # Auto-generate report
    if session.get("detections") and len(session["detections"]) > 0:
        session["report"] = generate_report(session)
    
    analyzed_count = sum(1 for r in results if r["success"])
    total_detections = sum(r["detections"] for r in results)
    
    logger.info(f"Sequential analysis complete: {analyzed_count}/{len(pending_photos)} photos, {total_detections} total detections")
    
    return {
        "status": "complete",
        "analyzed": analyzed_count,
        "total_photos": len(pending_photos),
        "total_detections": total_detections,
        "coverage": coverage,
        "cross_angle_score": cross_angle.get("overall_score", 0),
    }

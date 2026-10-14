"""
MANZAR — Cross-Angle Verification Service
=============================================
Compares YOLOv8 detections across multiple camera angles to determine
which evidence items are corroborated (seen by 2+ cameras) vs single-source.

Output: cross_angle_results dict stored in session.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("manzar.cross_angle")

# Minimum cameras to consider evidence "corroborated"
MIN_CORROBORATION = 2


def run_cross_angle_verification(photos: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Analyze detections across all photos and group by evidence class.
    
    Corroboration is based on INDEPENDENT OBSERVATIONS — either different
    devices OR different angles count as independent viewpoints.
    This means a single phone capturing from 4 angles still gets corroboration.
    
    Returns:
        {
            "overall_score": float (0-1),
            "evidence": [...],
            "summary": {...}
        }
    """
    if not photos:
        return {"overall_score": 0.0, "evidence": [], "summary": {}}

    # Group detections by class -> observation source (device:angle)
    class_sources: dict[str, dict[str, list]] = {}
    all_sources = set()  # unique "device:angle" combinations
    all_devices = set()
    all_angles = set()

    for photo in photos:
        device = photo.get("device_id", "unknown")
        angle = photo.get("angle_hint", "unknown") or "unknown"
        source_key = f"{device}:{angle}"
        all_sources.add(source_key)
        all_devices.add(device)
        all_angles.add(angle)

        detections = photo.get("detections", [])
        for det in detections:
            cls = det.get("class", "unknown")
            if cls not in class_sources:
                class_sources[cls] = {}
            if source_key not in class_sources[cls]:
                class_sources[cls][source_key] = []
            class_sources[cls][source_key].append(det)

    # Build evidence items
    evidence = []
    # Use the LARGER of device count or angle count as total viewpoints
    total_viewpoints = max(len(all_sources), 1)
    # For display, use device-based camera map if multiple devices, else angle-based
    multi_device = len(all_devices) > 1

    for cls, sources_map in class_sources.items():
        confirmed_count = len(sources_map)  # number of independent observations
        status = "corroborated" if confirmed_count >= MIN_CORROBORATION else "single-source" if confirmed_count == 1 else "unconfirmed"

        # Flatten all detections for this class
        all_dets = []
        for src_dets in sources_map.values():
            all_dets.extend(src_dets)

        # Build camera/source presence map
        cameras = {}
        if multi_device:
            # Multi-device: show device presence
            for dev in all_devices:
                has_it = any(s.startswith(f"{dev}:") for s in sources_map.keys())
                cameras[dev] = has_it
        else:
            # Single device: show angle presence (more useful for demo)
            for ang in sorted(all_angles):
                has_it = any(s.endswith(f":{ang}") for s in sources_map.keys())
                cameras[ang] = has_it

        # Determine which angles detected this evidence
        detected_angles = set()
        detected_devices = set()
        for src_key in sources_map.keys():
            parts = src_key.split(":", 1)
            detected_devices.add(parts[0])
            if len(parts) > 1:
                detected_angles.add(parts[1])

        evidence.append({
            "label": cls.replace("_", " ").title(),
            "type": cls,
            "cameras": cameras,
            "confirmed_by": confirmed_count,
            "total_cameras": total_viewpoints,
            "status": status,
            "max_confidence": max((d.get("confidence", 0) for d in all_dets), default=0),
            "angles": sorted(detected_angles),
            "devices": sorted(detected_devices),
            "detections": all_dets,
        })

    # Sort by confirmed_count descending, then by confidence
    evidence.sort(key=lambda x: (-x["confirmed_by"], -x["max_confidence"]))

    # Calculate overall consistency score
    if evidence:
        corroborated = sum(1 for e in evidence if e["status"] == "corroborated")
        score = corroborated / len(evidence)
    else:
        score = 0.0
        corroborated = 0

    result = {
        "overall_score": round(score, 3),
        "evidence": evidence,
        "summary": {
            "corroborated": corroborated,
            "single_source": len(evidence) - corroborated,
            "total_evidence_types": len(evidence),
            "total_viewpoints": total_viewpoints,
            "total_devices": len(all_devices),
            "total_angles": len(all_angles),
        },
    }

    logger.info(f"Cross-angle: {len(evidence)} evidence types, score={score:.2f}")
    return result

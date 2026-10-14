"""
MANZAR — Incident Report Generator
=====================================
Generates a structured accident investigation report matching the format
used by traffic police / forensic teams. Includes case metadata, evidence
findings, cross-angle verification, and forensic analysis.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("manzar.report")


def generate_report(session: dict[str, Any]) -> dict[str, Any]:
    """Generate a complete accident investigation report from session data."""

    session_id = session.get("session_id", "unknown")
    photos = session.get("photos", [])
    detections = session.get("detections", [])
    cross_angle = session.get("cross_angle_results", {})
    coverage = session.get("coverage_status", "unknown")
    missing = session.get("missing_angles", [])
    covered = session.get("covered_angles", [])
    reconstruction = session.get("reconstruction") or {}
    created = session.get("created_at", "")
    location = session.get("location", "Unknown Location")
    gps_lat = session.get("gps_lat", 0.0)
    gps_lng = session.get("gps_lng", 0.0)
    officer = session.get("officer_name", "Investigating Officer")
    badge = session.get("officer_badge", "N/A")
    case_number = session.get("case_number", f"MANZAR-{session_id}")
    timeline = session.get("timeline", [])

    # --- Case header ---
    case_header = {
        "case_number": case_number,
        "session_id": session_id,
        "officer_name": officer,
        "officer_badge": badge,
        "location": location,
        "gps": {"lat": gps_lat, "lng": gps_lng} if gps_lat and gps_lng else None,
        "date_initiated": created[:19].replace("T", " ") if created else "",
        "date_generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "photo_count": len(photos),
        "device_count": len(session.get("phones_joined", [])),
    }

    # --- Evidence findings ---
    findings = []
    class_dets: dict[str, list] = {}
    for d in detections:
        cls = d.get("class", "unknown")
        class_dets.setdefault(cls, []).append(d)

    severity_map = {
        "skid_mark": "HIGH",
        "vehicle_damage": "CRITICAL",
        "debris_field": "MEDIUM",
        "point_of_impact": "CRITICAL",
    }

    for cls, dets in class_dets.items():
        max_conf = max((d.get("confidence", 0) for d in dets), default=0)
        avg_conf = sum(d.get("confidence", 0) for d in dets) / len(dets) if dets else 0
        # Treat each unique device + angle combination as an independent
        # observation source. This means a single phone walking around the
        # scene still corroborates evidence when it is seen from multiple
        # angles, while multiple phones also count as independent sources.
        sources = list(set(
            f"{d.get('device_id', 'unknown')}:{d.get('angle_hint', 'unknown')}"
            for d in dets
        ))
        angles = list(set(d.get("angle_hint", "") for d in dets if d.get("angle_hint")))
        photos_with = list(set(d.get("source_photo", "") for d in dets))
        corroborated = len(sources) >= 2
        severity = severity_map.get(cls, "MEDIUM")

        # Detection source tracking
        yolo_dets = [d for d in dets if d.get("source") != "ai_inferred"]
        ai_dets = [d for d in dets if d.get("source") == "ai_inferred"]
        detection_source = "MANZAR"
        
        findings.append({
            "label": cls.replace("_", " ").title(),
            "type": cls,
            "severity": severity,
            "count": len(dets),
            "max_confidence": max_conf,
            "avg_confidence": avg_conf,
            "sources": sources,
            "angles": angles,
            "photos": photos_with,
            "corroborated": corroborated,
            "corroboration": f"Confirmed from {len(sources)} viewpoints" if corroborated else "Single viewpoint — needs verification",
            "status": "VERIFIED" if corroborated and max_conf >= 0.5 else "UNVERIFIED" if max_conf < 0.3 else "PROVISIONAL",
            "detection_source": detection_source,
            "yolo_count": len(yolo_dets),
            "ai_count": len(ai_dets),
        })

    findings.sort(key=lambda x: (
        {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}.get(x["severity"], 4),
        -x["max_confidence"],
    ))

    # --- Cross-angle summary ---
    overall_score = 0.0
    cross_summary = {}
    if isinstance(cross_angle, dict):
        overall_score = cross_angle.get("overall_score", 0.0)
        cross_summary = cross_angle.get("summary", {})

    # --- Investigation timeline ---
    timeline_entries = []
    for t in timeline:
        time_str = t.get("time", "")[:19].replace("T", " ") if t.get("time") else ""
        timeline_entries.append({
            "time": time_str,
            "event": t.get("event", ""),
            "detail": t.get("detail", ""),
            "icon": t.get("icon", ""),
        })

    # --- Forensic analysis ---
    forensic = _build_forensic_analysis(detections, photos, cross_angle, coverage)

    # --- Recommendations ---
    recommendations = _build_recommendations(
        missing, detections, overall_score, reconstruction, coverage, findings
    )

    # --- Chain of custody ---
    custody = []
    for p in photos:
        custody.append({
            "photo_number": p.get("photo_number", 0),
            "filename": p.get("filename", ""),
            "device": p.get("device_id", "unknown"),
            "angle": p.get("angle_hint", "unknown"),
            "uploaded_at": p.get("uploaded_at", ""),
            "hash": p.get("photo_hash", "N/A"),
            "detections": len(p.get("detections", [])),
        })

    # --- Possible cause of accident ---
    possible_cause = _build_possible_cause(detections, photos, cross_angle, forensic)

    # --- Officer notes comparison ---
    officer_notes = session.get("officer_notes", "")
    notes_comparison = _build_notes_comparison(officer_notes, detections, photos, forensic)

    # --- Build full report ---
    report = {
        "case_header": case_header,
        "summary": _build_summary(case_header, findings, forensic, coverage),
        "possible_cause": possible_cause,
        "evidence_findings": findings,
        "forensic_analysis": forensic,
        "officer_notes_comparison": notes_comparison,
        "cross_angle_verification": {
            "overall_score": overall_score,
            "summary": cross_summary,
            "reliability": "HIGH" if overall_score >= 0.8 else "MODERATE" if overall_score >= 0.5 else "LOW",
        },
        "coverage": {
            "status": coverage,
            "covered": covered,
            "missing": missing,
            "coverage_pct": f"{len(covered)}/4" if covered else "0/4",
        },
        "reconstruction": {
            "status": reconstruction.get("status", "not started"),
            "elapsed": reconstruction.get("elapsed_seconds", 0),
            "backend": reconstruction.get("backend", "local"),
            "mesh_available": reconstruction.get("mesh_path") is not None,
            "pointcloud_available": reconstruction.get("pointcloud_path") is not None,
        },
        "chain_of_custody": custody,
        "investigation_timeline": timeline_entries,
        "recommendations": recommendations,
        "disclaimer": "This report is generated from automated AI-assisted analysis. Fault attribution requires additional investigation by qualified personnel. Findings are advisory and do not constitute legal determination of liability.",
    }

    logger.info(f"Report: {case_number} — {len(findings)} findings, score={overall_score:.0%}")
    return report


def _build_summary(header, findings, forensic, coverage):
    """Build a descriptive, human-readable executive summary in plain English."""
    parts = []

    # Opening: case info
    parts.append(
        f"This investigation report covers case {header.get('case_number', 'N/A')}, "
        f"which was opened on {header.get('date_initiated', 'an unknown date')} "
        f"at {header.get('location', 'an unspecified location')}."
    )
    parts.append(
        f"The investigating officer, {header.get('officer_name', 'an unidentified officer')}, "
        f"collected {header.get('photo_count', 0)} photograph{'s' if header.get('photo_count', 0) != 1 else ''} "
        f"using {header.get('device_count', 0)} connected device{'s' if header.get('device_count', 0) != 1 else ''}."
    )

    # Evidence findings — explain each one in plain language
    if findings:
        evidence_lines = []
        for f in findings:
            label = f.get("label", "Unknown Evidence")
            conf_pct = round(f.get("max_confidence", 0) * 100)
            count = f.get("count", 0)
            status = f.get("status", "UNKNOWN")
            angles = f.get("angles", [])
            corroboration = f.get("corroboration", "")

            line = f"MANZAR detected **{label}** in {count} photo{'s' if count != 1 else ''} "
            line += f"with {conf_pct}% confidence"

            if status == "VERIFIED":
                line += f". This finding has been confirmed from multiple camera angles, making it highly reliable"
            elif status == "PROVISIONAL":
                line += f". This finding needs more photos from different angles to be fully confirmed"
            else:
                line += f". This finding has low confidence and should be verified by the officer on scene"

            if angles:
                line += f". Photos were taken from the {', '.join(angles)} view{'s' if len(angles) > 1 else ''}"

            line += "."
            evidence_lines.append(line)

        parts.append(" ".join(evidence_lines))
    else:
        parts.append(
            "No evidence was detected in the uploaded photographs. "
            "This could mean the photos do not clearly show accident-related damage, "
            "or the images may need to be retaken with better lighting and focus."
        )

    # Forensic analysis — explain in simple terms
    forensic_parts = []
    if forensic.get("speed_estimate"):
        forensic_parts.append(
            f"Based on the length of the skid marks found at the scene, "
            f"the vehicle was estimated to be traveling at approximately {forensic['speed_estimate']} "
            f"at the time of braking."
        )
    if forensic.get("impact_points"):
        forensic_parts.append(
            f"{forensic['impact_points']} point{'s' if forensic['impact_points'] != 1 else ''} of impact "
            f"{'were' if forensic['impact_points'] != 1 else 'was'} identified, "
            f"showing where the collision took place."
        )
    if forensic.get("vehicle_damage_count"):
        forensic_parts.append(
            f"Vehicle damage was clearly visible in the photographs "
            f"({forensic.get('vehicle_damage_confidence', 'high')} confidence level)."
        )
    if forensic.get("debris_detected"):
        forensic_parts.append(
            f"Scattered debris was also found at the scene, "
            f"further confirming that a significant collision occurred."
        )
    if forensic_parts:
        parts.append(" ".join(forensic_parts))

    # Coverage
    if coverage.upper() == "SUFFICIENT":
        parts.append(
            "The scene has been thoroughly photographed from all required angles (front, back, left, and right), "
            "providing a complete 360-degree view of the accident."
        )
    else:
        parts.append(
            f"Important note: The current photo coverage is incomplete ({coverage}). "
            f"Additional photographs from the remaining angles are needed to ensure no evidence is missed."
        )

    # Closing
    parts.append(
        "This report was automatically generated using AI-powered image analysis. "
        "All findings should be reviewed by the investigating officer before being used in official proceedings."
    )

    return " ".join(parts)


def _build_forensic_analysis(detections, photos, cross_angle, coverage):
    """Generate forensic analysis from evidence patterns."""
    analysis = {}

    # Count evidence types
    classes = {}
    for d in detections:
        cls = d.get("class", "unknown")
        classes.setdefault(cls, []).append(d)

    # Skid mark analysis
    if "skid_mark" in classes:
        skid_dets = classes["skid_mark"]
        max_conf = max(d.get("confidence", 0) for d in skid_dets)
        # Estimate skid length from bbox (rough approximation)
        if skid_dets and skid_dets[0].get("bbox"):
            bbox = skid_dets[0]["bbox"]
            pixel_length = abs(bbox[2] - bbox[0])
            # Rough: assume ~800px wide photo, car ~4m, so 200px/m
            est_meters = round(pixel_length / 200 * 0.8, 1)
            analysis["skid_mark_length"] = f"~{est_meters}m estimated"
            # Speed estimation: v = sqrt(2 * mu * g * d) where mu=0.7, g=9.81
            if est_meters > 0:
                speed_kmh = round(math.sqrt(2 * 0.7 * 9.81 * est_meters) * 3.6)
                analysis["speed_estimate"] = f"{speed_kmh}-{speed_kmh + 15} km/h"

    # Point of impact analysis
    if "point_of_impact" in classes:
        impact_count = len(classes["point_of_impact"])
        analysis["impact_points"] = impact_count
        analysis["impact_status"] = "Located" if impact_count > 0 else "Not detected"

    # Vehicle damage analysis
    if "vehicle_damage" in classes:
        damage_dets = classes["vehicle_damage"]
        max_conf = max(d.get("confidence", 0) for d in damage_dets)
        analysis["vehicle_damage_count"] = len(damage_dets)
        analysis["vehicle_damage_confidence"] = f"{max_conf:.0%}"
        # Check if damage from multiple angles
        damage_devices = set(d.get("device_id", "") for d in damage_dets)
        if len(damage_devices) >= 2:
            analysis["damage_corroboration"] = "Multi-angle confirmed"
            analysis["fault_indicator"] = "Consistent with collision damage pattern"

    # Debris analysis
    if "debris_field" in classes:
        analysis["debris_detected"] = True
        analysis["debris_count"] = len(classes["debris_field"])

    # Overall scene classification
    total_types = len(classes)
    if total_types >= 3:
        analysis["scene_complexity"] = "HIGH — multiple evidence types detected"
    elif total_types >= 1:
        analysis["scene_complexity"] = "MODERATE"
    else:
        analysis["scene_complexity"] = "LOW — insufficient evidence"

    # Build a clear, explanatory forensic summary in plain English
    summary_parts = []

    # Introduction: what the AI looked at
    summary_parts.append(
        "**Overview of Analysis:**\n\n"
        "MANZAR's AI system carefully examined each photograph uploaded to this case, looking for key signs of a traffic accident. "
        "The system searched for four main types of evidence: damage to vehicles, the exact point where vehicles collided, "
        "skid marks on the road, and scattered debris. Each piece of evidence tells a part of the story about what happened. "
        "Below is a detailed explanation of what was found and what it means for understanding this accident."
    )

    if "skid_mark_length" in analysis:
        summary_parts.append(
            f"**Skid Mark Analysis:**\n\n"
            f"The photographs show clear skid marks on the road surface, with an estimated length of "
            f"{analysis['skid_mark_length']}. Skid marks are created when a vehicle's tyres lock up during "
            f"hard braking, causing the rubber to leave dark streaks on the road. These marks are crucial evidence "
            f"because they show that the driver attempted to stop before the collision.\n\n"
            f"**Technical Details:** The length of skid marks helps investigators calculate the vehicle's speed "
            f"before braking. Using the standard physics formula (speed = square root of 2 × friction coefficient × "
            f"gravity × distance), and assuming a typical road friction coefficient of 0.7, we can estimate the vehicle's speed."
        )
        if "speed_estimate" in analysis:
            summary_parts.append(
                f"**Speed Estimation:**\n\n"
                f"Based on the skid mark length, the vehicle was likely traveling at approximately "
                f"{analysis['speed_estimate']} when the driver began braking. This calculation uses established "
            f"forensic engineering methods. However, this is an estimate — the actual speed could vary depending on "
            f"road conditions (wet or dry), tyre quality, road slope, and vehicle weight. For a precise speed "
            f"calculation, a detailed on-site investigation would be needed."
            )

    if "impact_points" in analysis:
        count = analysis['impact_points']
        summary_parts.append(
            f"**Point of Impact Analysis:**\n\n"
            f"MANZAR's AI identified {count} distinct point{'s' if count != 1 else ''} of impact in the photographs. "
            f"The point of impact is the exact location where two vehicles (or a vehicle and another object) made contact "
            f"during the collision. This is one of the most critical pieces of evidence in any accident investigation.\n\n"
            f"**Why It Matters:** The impact point reveals the direction of force, the angle of collision, and helps "
            f"reconstruct the sequence of events. For example, if the impact is on the front-left corner of one vehicle "
            f"and the front-right of another, it suggests a head-on or angled collision. Multiple impact points can "
            f"indicate a complex collision with secondary impacts."
        )

    if "vehicle_damage_count" in analysis:
        dmg_count = analysis['vehicle_damage_count']
        summary_parts.append(
            f"**Vehicle Damage Assessment:**\n\n"
            f"The AI detected visible damage in {dmg_count} distinct area{'s' if dmg_count != 1 else ''} across the "
            f"photographs, with a confidence level of {analysis.get('vehicle_damage_confidence', 'high')}. "
            f"Vehicle damage includes dents, crushed panels, broken lights, cracked windshields, and any deformation "
            f"to the vehicle's body structure.\n\n"
            f"**Technical Significance:** The pattern and location of damage tells investigators about the collision dynamics. "
            f"For instance, front-end damage suggests a head-on collision, while side damage indicates a T-bone or side-swipe impact. "
            f"The severity of deformation can also indicate the speed at impact — more severe crushing typically means higher speed."
        )
        if "damage_corroboration" in analysis:
            summary_parts.append(
                "**Multi-Angle Confirmation:**\n\n"
                "This damage was observed from multiple camera angles or devices, which significantly strengthens "
                "its reliability as evidence. When the same damage appears in photos taken from different positions, "
                "it confirms the finding is accurate and not an artifact of lighting or perspective. Multi-angle "
                "corroboration is a key principle in forensic investigation."
            )

    if analysis.get("debris_detected"):
        summary_parts.append(
            f"**Debris Field Analysis:**\n\n"
            f"Scattered debris was detected across {analysis.get('debris_count', 0)} "
            f"photo{'s' if analysis.get('debris_count', 0) != 1 else ''}. Debris includes broken glass, plastic fragments, "
            f"metal pieces, vehicle parts, and fluid leaks scattered at the accident scene.\n\n"
            f"**Forensic Value:** Debris patterns are like a fingerprint of the collision. The location where debris is "
            f"most concentrated often marks the point of maximum force application. The spread pattern can reveal the "
            f"direction of travel and the severity of impact. For example, glass scattered in a wide arc suggests a "
            f"high-energy impact, while a small, concentrated debris field indicates a lower-speed collision."
        )

    complexity = analysis.get("scene_complexity", "Unknown")
    if "HIGH" in complexity:
        summary_parts.append(
            "**Overall Scene Assessment:**\n\n"
            "This accident scene shows a HIGH level of evidence complexity, with multiple types of forensic indicators "
            "present — including vehicle damage, impact points, and possibly skid marks or debris. This comprehensive "
            "evidence profile allows investigators to build a detailed reconstruction of the accident sequence. "
            "The presence of multiple evidence types increases confidence in the findings and provides cross-verification "
            "of the collision dynamics."
        )
    elif "MODERATE" in complexity:
        summary_parts.append(
            "**Overall Scene Assessment:**\n\n"
            "The scene shows a MODERATE level of evidence. While some key indicators are present, additional photographs "
            "from different angles would strengthen the investigation. Specifically, photos from the missing angles "
            "(as noted in the coverage section) would provide a more complete picture of the accident and help confirm "
            "the findings through multi-angle corroboration."
        )
    else:
        summary_parts.append(
            "**Overall Scene Assessment:**\n\n"
            "The current photographs show LIMITED evidence for a complete forensic analysis. This could be due to "
            "poor image quality, insufficient coverage of the accident scene, or the photos not capturing the main "
            "damage areas. For a thorough investigation, it is recommended to upload clearer, closer photos that show "
            "the vehicle damage, impact points, and any road evidence like skid marks or debris."
        )

    analysis["summary"] = " ".join(summary_parts) if summary_parts else (
        "The uploaded photographs did not contain enough visible evidence for forensic analysis. "
        "Please ensure photos are well-lit, in focus, and capture the full extent of any damage."
    )

    return analysis


def _build_possible_cause(detections, photos, cross_angle, forensic):
    """Generate a descriptive paragraph about the possible cause of the accident."""
    classes = {}
    for d in detections:
        cls = d.get("class", "unknown")
        classes.setdefault(cls, []).append(d)

    has_damage = "vehicle_damage" in classes
    has_skid = "skid_mark" in classes
    has_debris = "debris_field" in classes
    has_impact = "point_of_impact" in classes
    speed_est = forensic.get("speed_estimate", "")
    damage_count = len(classes.get("vehicle_damage", []))
    debris_count = len(classes.get("debris_field", []))

    parts = []

    # Opening — set the scene based on evidence
    if has_damage and has_skid and has_debris:
        parts.append(
            "Based on the evidence collected at the scene, this accident appears to have involved "
            "a significant collision or loss of vehicle control. "
            "The combination of vehicle damage, skid marks on the road surface, and scattered debris "
            "paints a clear picture of what likely happened."
        )
    elif has_damage and has_skid:
        parts.append(
            "The evidence suggests this accident involved a collision preceded by emergency braking. "
            "The presence of both vehicle damage and skid marks on the road indicates the driver "
            "attempted to slow down before impact."
        )
    elif has_damage and has_debris:
        parts.append(
            "The scene shows signs of a direct collision, with visible damage to the vehicle and "
            "scattered debris across the road surface. The debris pattern suggests parts broke off "
            "at the moment of impact."
        )
    elif has_damage:
        parts.append(
            "The photographs show visible damage to the vehicle, consistent with a traffic collision. "
            "The extent and location of the damage can help determine the direction and severity of the impact."
        )
    else:
        parts.append(
            "The current evidence is limited, and a definitive cause cannot yet be established. "
            "Further investigation and additional photographs may reveal more about what happened."
        )

    # Speed analysis
    if speed_est:
        parts.append(
            f"The skid mark analysis estimates the vehicle was traveling at approximately {speed_est} "
            f"when the brakes were applied. "
            f"This speed, combined with road conditions at the time, may have contributed to the driver's "
            f"inability to stop in time to avoid the collision."
        )

    # Debris scatter
    if has_debris:
        parts.append(
            f"Debris was detected across {debris_count} area{'s' if debris_count != 1 else ''} of the scene, "
            f"which is consistent with a forceful impact that caused parts to break away from the vehicle. "
            f"The spread of the debris field can indicate the direction of travel at the moment of collision."
        )

    # Impact points
    if has_impact:
        impact_count = len(classes["point_of_impact"])
        parts.append(
            f"MANZAR detected {impact_count} point{'s' if impact_count != 1 else ''} of impact, "
            f"which helps pinpoint exactly where the vehicles or objects made contact. "
            f"This information is valuable for reconstructing the sequence of events."
        )

    # Multi-angle corroboration
    if isinstance(cross_angle, dict):
        score = cross_angle.get("overall_score", 0)
        if score >= 0.7:
            parts.append(
                "Multiple camera angles have independently confirmed the key evidence items, "
                "giving high confidence in this analysis. "
                "The cross-verification strengthens the reliability of these findings."
            )
        elif score > 0:
            parts.append(
                "Some evidence was confirmed from multiple viewpoints, though additional angles "
                "would help strengthen the conclusions further."
            )

    # Common causes based on evidence pattern
    if has_skid and speed_est:
        parts.append(
            "Common causes consistent with this evidence pattern include excessive speed, "
            "delayed braking reaction, wet or slippery road conditions, or an unexpected obstacle "
            "in the vehicle's path. "
            "However, determining the exact cause requires additional investigation including "
            "witness statements, vehicle mechanical inspection, and road condition assessment."
        )
    elif has_damage:
        parts.append(
            "Common causes consistent with this type of damage include collision with another vehicle, "
            "impact with a stationary object, or loss of control due to road conditions. "
            "The investigating officer should consider all these possibilities along with witness accounts."
        )

    # Closing disclaimer
    parts.append(
        "It is important to note that this analysis is based solely on photographic evidence processed "
        "by AI detection systems. The final determination of accident cause should be made by the "
        "investigating officer after considering all available information, including driver statements, "
        "witness accounts, vehicle inspection reports, and road condition data."
    )

    return " ".join(parts)


def _build_recommendations(missing, detections, score, reconstruction, coverage, findings):
    """Build actionable recommendations in plain English."""
    recs = []

    if missing:
        angle_names = ", ".join(missing)
        recs.append(
            f"The scene has not yet been photographed from the {angle_names} angle{'s' if len(missing) > 1 else ''}. "
            f"Please take additional photos from {'these directions' if len(missing) > 1 else 'this direction'} "
            f"to ensure complete coverage of the accident scene."
        )

    if not detections:
        recs.append(
            "The AI was unable to detect any accident-related evidence in the current photographs. "
            "Please retake the photos in better lighting, making sure the damage and surrounding "
            "scene are clearly visible and in focus."
        )

    if score < 0.5 and detections:
        recs.append(
            "Some evidence was only seen from a single camera angle. To make the findings more "
            "reliable and legally strong, take additional photos from different positions so that "
            "each piece of evidence can be confirmed from multiple viewpoints."
        )

    if reconstruction.get("status") == "reconstruction_failed":
        recs.append(
            "The 3D scene reconstruction could not be completed. This usually happens when photos "
            "don't overlap enough. Try taking more photos with 70%+ overlap between each shot."
        )

    if reconstruction.get("status") == "reconstruction_done":
        recs.append(
            "A 3D model of the accident scene is now available. You can view it in the dashboard "
            "to examine the scene from any angle and take measurements."
        )

    # Evidence-specific recommendations
    for f in findings:
        if f["status"] == "UNVERIFIED":
            recs.append(
                f"The '{f['label'].lower()}' finding has low confidence "
                f"({f['max_confidence']:.0%}). The investigating officer should personally inspect "
                f"this evidence at the scene to confirm whether it is genuine."
            )

    if not recs:
        recs.append(
            "All evidence has been captured, cross-verified from multiple angles, and fully documented. "
            "The investigation is ready for the officer's final review and FIR filing."
        )
    return recs


def _build_notes_comparison(officer_notes: str, detections: list, photos: list, forensic: dict) -> dict:
    """Compare officer's initial notes with AI findings and identify contradictions."""
    
    if not officer_notes or not officer_notes.strip():
        return {
            "notes_provided": False,
            "message": "No initial notes were provided by the investigating officer.",
            "contradictions": [],
            "corroborations": [],
        }
    
    # Extract key claims from officer notes
    notes_lower = officer_notes.lower()
    
    # Build AI findings summary
    ai_findings = {
        "vehicle_damage": len([d for d in detections if d.get("class") == "vehicle_damage"]),
        "point_of_impact": len([d for d in detections if d.get("class") == "point_of_impact"]),
        "skid_marks": len([d for d in detections if d.get("class") == "skid_mark"]),
        "debris": len([d for d in detections if d.get("class") == "debris_field"]),
        "total_detections": len(detections),
        "photos_analyzed": len(photos),
    }
    
    contradictions = []
    corroborations = []
    
    # Check for contradictions
    # Example: Officer says "no damage" but AI found damage
    if "no damage" in notes_lower or "not damaged" in notes_lower:
        if ai_findings["vehicle_damage"] > 0:
            contradictions.append({
                "officer_claim": "Officer reported no vehicle damage",
                "ai_finding": f"AI detected {ai_findings['vehicle_damage']} area(s) of vehicle damage",
                "severity": "HIGH",
                "recommendation": "Physical inspection recommended to verify actual damage extent",
            })
    
    # Officer says "minor accident" but AI found significant evidence
    if "minor" in notes_lower and (ai_findings["vehicle_damage"] >= 3 or ai_findings["point_of_impact"] >= 2):
        contradictions.append({
            "officer_claim": "Officer described accident as minor",
            "ai_finding": f"AI found extensive evidence: {ai_findings['total_detections']} detections across multiple categories",
            "severity": "MEDIUM",
            "recommendation": "Review AI findings - accident may be more severe than initially assessed",
        })
    
    # Officer mentions specific damage location
    if "front" in notes_lower and "rear" not in notes_lower:
        rear_damage = len([d for d in detections if d.get("class") == "vehicle_damage" and d.get("angle_hint") == "rear"])
        if rear_damage > 0:
            contradictions.append({
                "officer_claim": "Officer only mentioned front damage",
                "ai_finding": f"AI also detected rear damage in {rear_damage} photo(s)",
                "severity": "MEDIUM",
                "recommendation": "Inspect rear of vehicle - additional damage may be present",
            })
    
    # Check for corroborations
    if "skid marks" in notes_lower or "brake marks" in notes_lower:
        if ai_findings["skid_marks"] > 0:
            corroborations.append({
                "officer_claim": "Officer noted skid/brake marks",
                "ai_finding": f"AI confirmed skid marks in {ai_findings['skid_marks']} photo(s)",
                "status": "CORROBORATED",
            })
    
    if "debris" in notes_lower or "glass" in notes_lower:
        if ai_findings["debris"] > 0:
            corroborations.append({
                "officer_claim": "Officer noted debris at scene",
                "ai_finding": f"AI detected debris in {ai_findings['debris']} photo(s)",
                "status": "CORROBORATED",
            })
    
    if "impact" in notes_lower or "collision point" in notes_lower:
        if ai_findings["point_of_impact"] > 0:
            corroborations.append({
                "officer_claim": "Officer identified impact point",
                "ai_finding": f"AI located {ai_findings['point_of_impact']} point(s) of impact",
                "status": "CORROBORATED",
            })
    
    # Build summary
    if not contradictions and not corroborations:
        summary = "The officer's notes have been recorded. No specific contradictions or corroborations were identified by the AI analysis."
    elif contradictions:
        summary = f"AI analysis identified {len(contradictions)} potential contradiction(s) between officer notes and photographic evidence. Review recommended."
    else:
        summary = f"AI analysis corroborated {len(corroborations)} aspect(s) of the officer's initial observations."
    
    return {
        "notes_provided": True,
        "officer_notes": officer_notes,
        "summary": summary,
        "ai_findings_summary": ai_findings,
        "contradictions": contradictions,
        "corroborations": corroborations,
        "total_contradictions": len(contradictions),
        "total_corroborations": len(corroborations),
    }

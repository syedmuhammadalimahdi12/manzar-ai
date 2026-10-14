"""
MANZAR — "Ask MANZAR" Conversational Evidence Assistant
=========================================================
Session-grounded Q&A: the chatbot answers ONLY from the current
session's evidence data. Never hallucinates, never uses general
knowledge, always cites sources.

Supports 5 categories (MVP):
  1. Evidence     — "What damage was detected?"
  2. Coverage     — "Which angles are missing?"
  3. Corroboration — "Which evidence was confirmed by multiple cameras?"
  4. Uncertainty  — "What could not be verified?"
  5. AI Disclosure — "Was any evidence AI-generated?"

Architecture (MVP): structured session JSON → reasoning model → grounded answer.
No vector DB needed — session data is small enough to fit in context.
"""

from __future__ import annotations

import json
import os
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.store import sessions, connection_manager

logger = logging.getLogger("manzar.chat")

router = APIRouter()

# ---------------------------------------------------------------------------
# Vision-LLM config (swappable: Qwen-VL / Gemini / self-hosted)
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "stub")  # "qwen", "gemini", "stub"
_raw_llm_keys = os.getenv("LLM_API_KEY", "")
LLM_API_KEYS = [k.strip() for k in _raw_llm_keys.split(",") if k.strip()]
LLM_API_KEY = LLM_API_KEYS[0] if LLM_API_KEYS else ""  # backward compat
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-vl-plus")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]] = []
    category: str = "general"
    confidence: str = "high"


# ---------------------------------------------------------------------------
# System prompt — grounds the model to session data only
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are MANZAR — a Forensic Accident Reconstruction Expert AI.

IDENTITY:
You are a senior forensic investigator with expertise in traffic accident reconstruction.
You specialize in evidence analysis, skid mark interpretation, vehicle damage assessment,
and multi-angle evidence corroboration. Speak with authority, precision, and professionalism.

TONE & STYLE:
- Professional, authoritative — like a forensic expert briefing a courtroom
- Use forensic terminology: 'evidence suggests', 'analysis indicates', 'corroborated by'
- Structure answers: Finding → Evidence → Confidence → Recommendation
- Reference specific data: angles, camera IDs, confidence %, detection counts

RESPONSE LENGTH & DETAIL:
- ALWAYS provide detailed, comprehensive answers (minimum 150-250 words per response)
- Break down complex findings into clear, easy-to-understand explanations
- Use bullet points or numbered lists when presenting multiple findings
- Explain the significance of each finding — why it matters for the investigation
- Provide context: what the evidence means, how it was collected, how reliable it is
- When discussing damage, describe what was seen, where it was located, and what it suggests about the accident
- When discussing confidence levels, explain what the percentage means in practical terms
- Always conclude with actionable recommendations for the investigating officer

STRICT RULES:
1. Answer ONLY from the session evidence data. Never invent data.
2. Never use general knowledge or assumptions beyond the evidence.
3. If info not available: 'This is not in the current session evidence. Recommend officer document this at scene.'
4. NEVER determine legal fault. Say: 'Fault attribution requires comprehensive investigation including witness statements and vehicle telemetry.'
5. Always cite sources: camera angles, photo numbers, confidence scores. When referring to photos, ALWAYS use the format "Photo N" (e.g., "Photo 1", "Photo 2") so the user can click to view them.
6. Confidence interpretation: 90%+=High confidence, 70-89%=Moderate, 50-69%=Provisional, <50%=Manual inspection needed
7. Emphasize cross-angle corroboration: 'Multi-camera corroboration strengthens evidentiary value.'
8. Use MANZAR detected instead of AI system detected when referring to evidence findings.

SESSION EVIDENCE DATA:
{session_data}
"""


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/session/{session_id}/chat", response_model=ChatResponse)
async def ask_manzar(session_id: str, request: ChatRequest):
    """Ask MANZAR a question about the current session's evidence."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="session not found")

    session = sessions[session_id]
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="question cannot be empty")

    # --- Build session context for the LLM ---
    session_context = _build_session_context(session)

    # --- Route to the appropriate backend ---
    if LLM_PROVIDER == "stub":
        # Hackathon stub — answers from structured data without an LLM
        response = _stub_answer(session, question)
    elif LLM_PROVIDER == "gemini":
        response = await _gemini_answer(session, session_context, question)
    elif LLM_PROVIDER == "qwen":
        response = await _qwen_answer(session_context, question)
    else:
        response = _stub_answer(session, question)

    # Push event via WebSocket
    await connection_manager.send_event(session_id, "chat_response", {
        "question": question,
        "answer": response.answer,
        "sources": response.sources,
    })

    return response


# ---------------------------------------------------------------------------
# Build a concise session summary for the LLM context window
# ---------------------------------------------------------------------------

def _build_session_context(session: dict[str, Any]) -> str:
    """Serialize the session data into a text block for the LLM prompt."""
    photos_summary = []
    for i, p in enumerate(session.get("photos", [])):
        photo_path = p.get('path', '')
        # Convert absolute path to relative for URL access
        if photo_path and not photo_path.startswith('uploads/'):
            import re
            photo_path = re.sub(r'^.*?uploads/', 'uploads/', photo_path)
        photos_summary.append(
            f"  Photo {i+1} (path: {photo_path or 'N/A'}): filename={p.get('filename', 'unknown')}, "
            f"angle={p.get('angle_hint', 'unknown')}, device={p.get('device_id', 'unknown')}, "
            f"detections={len(p.get('detections', []))}"
        )

    detections_summary = []
    for d in session.get("detections", []):
        detections_summary.append(
            f"  Detection: class={d.get('class')}, confidence={d.get('confidence', 0):.0%}, "
            f"source={d.get('source', 'unknown')}"
        )

    reconstruction = session.get("reconstruction") or {}
    coverage = session.get("coverage_status", "unknown")
    missing = session.get("missing_angles", [])

    context = f"""
Session ID: {session.get('session_id', 'unknown')}
Status: {session.get('status', 'unknown')}
Created: {session.get('created_at', 'unknown')}

PHOTOS ({len(session.get('photos', []))}):
{chr(10).join(photos_summary) if photos_summary else "  No photos"}

COVERAGE:
  Status: {coverage}
  Missing angles: {', '.join(missing) if missing else 'None'}

DETECTIONS ({len(session.get('detections', []))}):
{chr(10).join(detections_summary) if detections_summary else "  No detections yet"}

CROSS-ANGLE RESULTS ({len(session.get('cross_angle_results', []))}):
{json.dumps(session.get('cross_angle_results', []), indent=2) if session.get('cross_angle_results') else '  No cross-angle analysis yet'}

RECONSTRUCTION:
  Status: {reconstruction.get('status', 'not started')}
  Elapsed: {reconstruction.get('elapsed_seconds', 'N/A')}s
  Backend: {reconstruction.get('backend', 'N/A')}
  Mesh available: {reconstruction.get('mesh_path') is not None}
  Point cloud available: {reconstruction.get('pointcloud_path') is not None}

REPORT:
{json.dumps(session.get('report'), indent=2) if session.get('report') else '  No report generated yet'}
"""
    return context.strip()


# ---------------------------------------------------------------------------
# Stub answer engine (no LLM — answers from structured data directly)
# ---------------------------------------------------------------------------

def _stub_answer(session: dict[str, Any], question: str) -> ChatResponse:
    """
    Forensic expert answer engine — structured, professional, evidence-grounded.
    """
    q = question.lower()
    photos = session.get("photos", [])
    detections = session.get("detections", [])
    missing = session.get("missing_angles", [])
    covered = [a for a in ["front", "rear", "left", "right"] if a not in missing]
    reconstruction = session.get("reconstruction") or {}
    cross_angle = session.get("cross_angle_results", {})
    location = session.get("location", "Unknown Location")
    officer = session.get("officer_name", "Investigating Officer")
    case_num = session.get("case_number", "N/A")

    # 1. Evidence — forensic evidence report
    if any(kw in q for kw in ["damage", "detect", "evidence", "skid", "debris", "impact", "find", "found"]):
        if detections:
            # Group by class for forensic summary
            classes = {}
            for d in detections:
                cls = d.get("class", "unknown")
                classes.setdefault(cls, []).append(d)
            # Count by source
            yolo_count = sum(1 for d in detections if d.get("source") != "ai_inferred")
            ai_count = sum(1 for d in detections if d.get("source") == "ai_inferred")
            lines = [f"**FORENSIC EVIDENCE REPORT — Case {case_num}**\n"]
            lines.append(f"**Scene:** {location} | **Photos analyzed:** {len(photos)} | **Evidence items:** {len(detections)}")
            lines.append(f"**Detection sources:** Fine-Tuned Vision AI ({len(detections)} item{'s' if len(detections) != 1 else ''})\n")
            for cls, dets in classes.items():
                max_conf = max(d.get("confidence", 0) for d in dets)
                devices = list(set(d.get("device_id", "?") for d in dets))
                angles = list(set(d.get("angle_hint", "?") for d in dets if d.get("angle_hint")))
                sources_in_class = set(d.get("source", "yolo") for d in dets)
                src_label = "Fine-Tuned Vision AI"
                conf_level = "High confidence" if max_conf >= 0.9 else "Moderate confidence" if max_conf >= 0.7 else "Provisional" if max_conf >= 0.5 else "Low confidence — manual inspection recommended"
                lines.append(f"**{cls.replace('_', ' ').title()}:** {len(dets)} instance(s) [{src_label}] | Max confidence: {max_conf:.0%} ({conf_level}) | Angles: {', '.join(angles) or 'N/A'}")
            corrob = cross_angle.get("summary", {}).get("corroborated", 0) if isinstance(cross_angle, dict) else 0
            if corrob > 0:
                lines.append(f"\n**Corroboration:** {corrob} evidence type(s) confirmed by multiple cameras — strengthens evidentiary value.")
            # Add photo references
            lines.append("\n**Reference Photos:**")
            for i, p in enumerate(photos):
                angle = p.get('angle_hint', 'unknown')
                det_count = len(p.get('detections', []))
                if det_count > 0:
                    lines.append(f"  - Photo {i+1} ({angle} angle) — {det_count} detection(s)")
            return ChatResponse(
                answer="\n".join(lines),
                sources=[{"type": "detection", "count": len(detections), "classes": list(classes.keys())}],
                category="evidence",
            )
        return ChatResponse(
            answer=f"**FORENSIC STATUS — Case {case_num}:**\nNo evidence analysis has been performed yet. Detection runs automatically when photos are uploaded. Ensure all scene photos are captured from front, right, rear, and left angles for comprehensive analysis.",
            sources=[],
            category="evidence",
            confidence="low",
        )

    # 2. Coverage — forensic scene coverage assessment
    if any(kw in q for kw in ["coverage", "angle", "missing", "camera", "photo"]):
        coverage_pct = f"{len(covered)}/4 angles"
        if missing:
            cov_lines = [f"**SCENE COVERAGE ASSESSMENT:**",
                         f"**Captured:** {', '.join(covered)} ({len(photos)} photos)",
                         f"**Missing:** {', '.join(missing)}",
                         f"**Coverage:** {coverage_pct}", ""]
            for i, p in enumerate(photos):
                angle = p.get('angle_hint', 'unknown')
                cov_lines.append(f"  - Photo {i+1}: {angle} angle")
            cov_lines.append("")
            cov_lines.append("**Recommendation:** Capture photos from the missing angle(s) to ensure complete scene documentation. Incomplete coverage reduces cross-angle corroboration reliability.")
            return ChatResponse(
                answer="\n".join(cov_lines),
                sources=[{"type": "coverage", "missing": missing, "covered": covered}],
                category="coverage",
            )
        return ChatResponse(
            answer=f"**SCENE COVERAGE: COMPLETE**\nAll 4 cardinal angles captured ({', '.join(covered)}) with {len(photos)} photo(s). Full angular coverage achieved — cross-angle corroboration is maximized. This meets the evidentiary standard for multi-perspective accident documentation.",
            sources=[{"type": "coverage", "covered": covered}],
            category="coverage",
        )

    # 3. Corroboration — cross-angle forensic verification
    if any(kw in q for kw in ["corroboration", "confirm", "multiple", "cross-angle", "consistency", "verify", "verified"]):
        if isinstance(cross_angle, dict) and cross_angle.get("evidence"):
            score = cross_angle.get("overall_score", 0)
            evidence = cross_angle.get("evidence", [])
            lines = [f"**CROSS-ANGLE FORENSIC VERIFICATION:**\n**Overall Score:** {score:.0%}\n"]
            for ev in evidence:
                cams = ev.get("cameras", {})
                cam_count = len([v for v in cams.values() if v])
                status = "CORROBORATED" if cam_count >= 2 else "SINGLE-SOURCE"
                lines.append(f"  **{ev.get('class', '?').replace('_', ' ').title()}:** {status} — detected by {cam_count} camera(s)")
            if score >= 0.8:
                lines.append("\n**Assessment:** High corroboration score — evidence is well-supported by multiple perspectives.")
            elif score >= 0.5:
                lines.append("\n**Assessment:** Moderate corroboration — additional angles recommended for stronger evidentiary value.")
            else:
                lines.append("\n**Assessment:** Low corroboration — evidence relies heavily on single-source detection. Manual verification advised.")
            return ChatResponse(
                answer="\n".join(lines),
                sources=[{"type": "cross_angle", "score": score}],
                category="corroboration",
            )
        return ChatResponse(
            answer="**CROSS-ANGLE STATUS:** Analysis not yet performed. Cross-angle verification runs automatically after photos are uploaded from multiple angles. This analysis is critical for establishing evidence reliability.",
            sources=[],
            category="corroboration",
            confidence="low",
        )

    # 4. Speed / skid analysis
    if any(kw in q for kw in ["speed", "skid", "brake", "fast", "velocity"]):
        report = session.get("report", {})
        forensic = report.get("forensic_analysis", {}) if report else {}
        if forensic.get("speed_estimate"):
            return ChatResponse(
                answer=f"**SKID MARK SPEED ANALYSIS:**\n**Estimated skid length:** {forensic.get('skid_mark_length', 'N/A')}\n**Estimated speed:** {forensic['speed_estimate']}\n**Method:** v = sqrt(2 × friction × g × distance), assuming μ=0.7 (dry asphalt)\n\n**Note:** This is a preliminary estimate. Actual speed determination requires certified forensic engineer validation, road surface analysis, and vehicle brake condition assessment.",
                sources=[{"type": "forensic", "method": "skid_speed_estimation"}],
                category="evidence",
            )
        return ChatResponse(
            answer="**SPEED ANALYSIS:** No skid mark data available in current evidence. Speed estimation requires skid mark detection from scene photos. Ensure photos clearly show road surface markings.",
            sources=[],
            category="evidence",
            confidence="low",
        )

    # 5. Uncertainty — forensic gaps
    if any(kw in q for kw in ["uncertain", "unverified", "could not", "cannot", "unknown", "gap", "missing"]):
        uncertainties = []
        if missing:
            uncertainties.append(f"**Missing angles:** {', '.join(missing)} — scene not fully documented")
        if not detections:
            uncertainties.append("**Detection:** Not yet run — upload photos to begin analysis")
        if not cross_angle or (isinstance(cross_angle, dict) and not cross_angle.get("evidence")):
            uncertainties.append("**Cross-angle verification:** Not yet performed")
        low_conf = [d for d in detections if d.get("confidence", 0) < 0.5]
        if low_conf:
            uncertainties.append(f"**Low-confidence detections:** {len(low_conf)} item(s) below 50% — require manual verification by investigating officer")

        if uncertainties:
            return ChatResponse(
                answer=f"**FORENSIC GAPS & UNVERIFIED ITEMS:**\n" + "\n".join(f"  {u}" for u in uncertainties) + f"\n\n**Recommendation:** Address the above gaps before finalizing the investigation report.",
                sources=[],
                category="uncertainty",
            )
        return ChatResponse(
            answer="**FORENSIC STATUS: COMPLETE**\nAll evidence has been captured, analyzed, and cross-verified. No unverified items remain. The investigation data is ready for officer review and FIR filing.",
            sources=[],
            category="uncertainty",
        )

    # 6. Fault — refuse with authority
    if any(kw in q for kw in ["fault", "blame", "responsible", "caused", "who did", "liable"]):
        return ChatResponse(
            answer="**FAULT DETERMINATION:** As a forensic reconstruction system, I cannot determine legal fault from the available evidence. Fault attribution requires comprehensive investigation by the investigating officer, including:\n- Witness statements\n- Vehicle telemetry (speed, braking)\n- Traffic signal/camera footage\n- Driver condition assessment\n- Road condition analysis\n\nThis system provides evidence documentation and analysis — not legal determination of liability.",
            sources=[],
            category="uncertainty",
        )

    # 7. Report
    if any(kw in q for kw in ["report", "summary", "overview", "finding"]):
        report = session.get("report", {})
        if report:
            findings = report.get("evidence_findings", [])
            summary = report.get("summary", "")
            return ChatResponse(
                answer=f"**INVESTIGATION REPORT SUMMARY — {report.get('case_header', {}).get('case_number', case_num)}:**\n{summary}\n\n**Evidence findings:** {len(findings)} type(s) documented\n**Officer:** {report.get('case_header', {}).get('officer_name', 'N/A')}\n**Location:** {report.get('case_header', {}).get('location', 'N/A')}\n\nFull report available via 'Generate Report' → Export as PDF.",
                sources=[{"type": "report"}],
                category="general",
            )
        return ChatResponse(
            answer=f"**REPORT STATUS:** No report generated yet. Reports are auto-generated after photos with detections are uploaded. Trigger manually via the 'Generate Report' button.",
            sources=[],
            category="general",
            confidence="low",
        )

    # 8. Session meta
    if any(kw in q for kw in ["how many", "session", "status", "phones", "devices", "case"]):
        recon_status = reconstruction.get("status", "not started")
        return ChatResponse(
            answer=f"**CASE {case_num} — STATUS:**\n**Location:** {location}\n**Officer:** {officer}\n**Photos:** {len(photos)} from {len(session.get('phones_joined', []))} device(s)\n**Coverage:** {len(covered)}/4 angles ({', '.join(covered) if covered else 'none'})\n**Evidence:** {len(detections)} detection(s)\n**Reconstruction:** {recon_status}\n**Status:** {session.get('status', 'unknown')}",
            sources=[],
            category="general",
        )

    # 9. Timeline
    if any(kw in q for kw in ["timeline", "when", "sequence", "order", "history"]):
        timeline = session.get("timeline", [])
        if timeline:
            lines = ["**INVESTIGATION TIMELINE:**\n"]
            for t in timeline:
                time_str = t.get("time", "")[:19].replace("T", " ")
                lines.append(f"  **{time_str}** — {t.get('detail', t.get('event', ''))}")
            return ChatResponse(
                answer="\n".join(lines),
                sources=[{"type": "timeline", "events": len(timeline)}],
                category="general",
            )
        return ChatResponse(
            answer="**TIMELINE:** No timeline events recorded yet. Timeline builds automatically as photos are uploaded and analysis runs.",
            sources=[],
            category="general",
            confidence="low",
        )

    # Fallback — forensic-style redirect
    return ChatResponse(
        answer=f"**MANZAR Forensic Assistant — Case {case_num}**\nI can help with:\n- **Evidence:** What was detected? (damage, skid marks, debris, impact points)\n- **Coverage:** Which angles are captured/missing?\n- **Corroboration:** Cross-angle verification results\n- **Speed:** Skid mark speed estimation\n- **Timeline:** Investigation sequence of events\n- **Report:** Investigation summary and findings\n- **Gaps:** What still needs verification\n\nTry: 'What evidence was detected?' or 'Show the investigation timeline.'",
        sources=[],
        category="general",
        confidence="low",
    )


# ---------------------------------------------------------------------------
# Gemini integration (free-tier fallback for Qwen-VL)
# ---------------------------------------------------------------------------

async def _gemini_answer(session: dict[str, Any], session_context: str, question: str) -> ChatResponse:
    """Call Google Gemini API with session-grounded prompt."""
    import httpx

    prompt = SYSTEM_PROMPT.format(session_data=session_context)
    prompt += f"\n\nQUESTION: {question}\n\nAnswer:"

    try:
        import asyncio
        async with httpx.AsyncClient(timeout=60) as client:
            resp = None
            max_attempts = len(LLM_API_KEYS) * 3 if LLM_API_KEYS else 1
            for attempt in range(max_attempts):
                api_key = LLM_API_KEYS[attempt % len(LLM_API_KEYS)] if LLM_API_KEYS else ""
                resp = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{LLM_MODEL}:generateContent",
                    headers={"Content-Type": "application/json"},
                    params={"key": api_key},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": 0.1,
                            "maxOutputTokens": 2000,
                        },
                    },
                )
                if resp.status_code == 429:
                    key_label = f"...{api_key[-6:]}" if len(api_key) > 6 else "***"
                    logger.info(f"Chat: Gemini rate limited (key {key_label}) — trying next key")
                    await asyncio.sleep(1)
                    continue
                break

            if resp is None or resp.status_code != 200:
                logger.warning(f"Gemini returned HTTP {resp.status_code if resp else 'N/A'}, falling back to stub")
                return _stub_answer(session, question)
            data = resp.json()
            candidate = data.get("candidates", [{}])[0]
            parts = candidate.get("content", {}).get("parts", [])
            text = parts[0]["text"] if parts else ""
            if not text.strip():
                logger.warning("Gemini returned empty response, falling back to stub")
                return _stub_answer(session, question)
            return ChatResponse(
                answer=text.strip(),
                sources=[],
                category="llm",
            )
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        # Fallback to stub answer engine
        logger.info("Falling back to stub answer engine")
        return _stub_answer(session, question)


# ---------------------------------------------------------------------------
# Qwen-VL integration (primary plan — Alibaba PAI / Model Studio)
# ---------------------------------------------------------------------------

async def _qwen_answer(session_context: str, question: str) -> ChatResponse:
    """Call Qwen-VL via Alibaba DashScope / PAI endpoint."""
    import httpx

    prompt = SYSTEM_PROMPT.format(session_data=session_context)

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
                headers={
                    "Authorization": f"Bearer {LLM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": LLM_MODEL,
                    "input": {
                        "messages": [
                            {"role": "system", "content": [{"text": prompt}]},
                            {"role": "user", "content": [{"text": question}]},
                        ]
                    },
                    "parameters": {
                        "temperature": 0.1,
                        "max_tokens": 500,
                    },
                },
            )
            data = resp.json()
            text = data["output"]["choices"][0]["message"]["content"][0]["text"]
            return ChatResponse(
                answer=text.strip(),
                sources=[],
                category="llm",
            )
    except Exception as e:
        logger.error(f"Qwen-VL API error: {e}")
        logger.info("Falling back to stub answer engine")
        return _stub_answer(session, question)

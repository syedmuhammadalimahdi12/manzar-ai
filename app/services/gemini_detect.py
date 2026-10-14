"""
MANZAR — Gemini Vision Detection
==================================
Uses Reasioning models's vision capabilities to identify accident scene evidence
that the YOLO model might miss (skid marks, debris, impact points).

Detections from Gemini are marked as source="ai_inferred" to distinguish
them from YOLO detections (source="yolo").
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from pathlib import Path
from typing import Any

try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False

logger = logging.getLogger("manzar.gemini_detect")

# Support multiple API keys (comma-separated) for rate-limit fallback
_raw_keys = os.getenv("LLM_API_KEY", "")
GEMINI_API_KEYS = [k.strip() for k in _raw_keys.split(",") if k.strip()]
GEMINI_API_KEY = GEMINI_API_KEYS[0] if GEMINI_API_KEYS else ""  # backward compat
GEMINI_MODEL = os.getenv("LLM_MODEL", "gemini-3.6-flash")
_KEY_INDEX = 0  # round-robin index for key selection

# OpenRouter configuration (alternative provider with free Qwen models)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen3-vl-8b-instruct")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

def _get_next_key():
    """Get next API key in round-robin order."""
    global _KEY_INDEX
    if not GEMINI_API_KEYS:
        return ""
    key = GEMINI_API_KEYS[_KEY_INDEX % len(GEMINI_API_KEYS)]
    _KEY_INDEX += 1
    return key

def _rotate_key():
    """Rotate to next key (called on 429 rate limit)."""
    global _KEY_INDEX
    _KEY_INDEX += 1
    return GEMINI_API_KEYS[_KEY_INDEX % len(GEMINI_API_KEYS)] if GEMINI_API_KEYS else ""

# Determine which provider to use
VISION_PROVIDER = os.getenv("VISION_PROVIDER", "gemini")  # "gemini" or "openrouter"
if VISION_PROVIDER == "openrouter" and OPENROUTER_API_KEY:
    logger.info(f"Vision provider: OpenRouter (model={OPENROUTER_MODEL})")
else:
    logger.info(f"Gemini configured with {len(GEMINI_API_KEYS)} API key(s), model={GEMINI_MODEL}")



def _resize_for_gemini(image_path: str | Path, max_size: int = 1024) -> bytes:
    """Resize image to reduce API payload size and latency. Keep aspect ratio."""
    image_path = Path(image_path)
    if not HAS_PIL:
        return image_path.read_bytes()
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            if w > max_size or h > max_size:
                ratio = min(max_size / w, max_size / h)
                new_size = (int(w * ratio), int(h * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            return buf.getvalue()
    except Exception as e:
        logging.getLogger("manzar.gemini_detect").warning(f"Image resize failed: {e}, using original")
        return image_path.read_bytes()

DETECTION_PROMPT = """You are analyzing a traffic accident scene photograph for forensic evidence.

Look carefully at this image and identify ALL visible accident scene evidence.
Focus specifically on these evidence types:
- skid_mark: Tire skid marks, brake marks, or tire drag marks on road surface
- debris_field: Scattered debris — broken glass, plastic fragments, metal pieces, vehicle parts
- point_of_impact: The impact point — collision zone, crushed area, contact point between vehicles
- vehicle_damage: Damage to vehicles — dents, crumple zones, broken panels

For each piece of evidence you identify, provide a JSON object with:
- "class": one of [skid_mark, debris_field, point_of_impact, vehicle_damage]
- "confidence": float between 0.0 and 1.0 (how confident you are)
- "description": brief description of what you see
- "location": approximate position in image [x_center, y_center, width, height] as fractions 0-1

Return ONLY a JSON array of detections. If no evidence is visible, return [].

IMPORTANT: Only report evidence you can actually SEE in the image. Do not invent evidence."""

COMBINED_PROMPT = """You are a forensic AI analyzing traffic accident scene photographs with extreme attention to detail.

First, decide if this image is relevant to a traffic accident. Consider it RELEVANT if it shows: damaged vehicles, road debris, skid marks, accident scene with emergency vehicles/cones, or vehicle collision damage.

Then, if relevant, identify ALL visible accident evidence from these classes. Be THOROUGH and look for even subtle signs of damage:

- skid_mark: Tire skid marks, brake marks, or tire drag marks on road surface (dark streaks, rubber marks)
- debris_field: Scattered debris — broken glass, plastic fragments, metal pieces, vehicle parts, fluid leaks
- point_of_impact: The impact point — collision zone, crushed area, contact point between vehicles, where damage is most severe
- vehicle_damage: ANY damage to vehicles including:
  * Dents, dents, dents (even small ones)
  * Crumpled or crushed body panels
  * Broken or cracked bumpers, headlights, taillights
  * Misaligned panels, gaps between panels
  * Scratches, scrapes, paint damage
  * Broken windows, mirrors
  * ANY visible deformation or damage to the vehicle body

Look at ALL parts of the vehicle visible in the image — front, rear, sides, top. Damage can appear on any angle. Even if the damage seems minor, report it.

Also determine the camera angle of the photograph. Choose the ONE best match from: front, rear, left, right.
- front: camera is facing the front of the damaged vehicle(s) (headlights, grille, hood visible)
- rear: camera is facing the back/tail of the damaged vehicle(s) (taillights, trunk, rear bumper visible)
- left: camera is on the left side of the damaged vehicle(s) (driver side in left-hand drive countries)
- right: camera is on the right side of the damaged vehicle(s) (passenger side in left-hand drive countries)
If the angle cannot be determined, use "unknown".

Return ONLY this JSON structure:
{
  "relevant": true/false,
  "reason": "brief explanation",
  "angle": "front | rear | left | right | unknown",
  "detections": [
    {"class": "one of [skid_mark, debris_field, point_of_impact, vehicle_damage]", "confidence": 0.0-1.0, "description": "brief description of what you see", "bbox": [x1, y1, x2, y2]}
  ]
}

CRITICAL: The "bbox" field MUST be [x1, y1, x2, y2] format (NOT center+width+height!):
- x1, y1 = TOP-LEFT corner of the box (0.0 to 1.0)
- x2, y2 = BOTTOM-RIGHT corner of the box (0.0 to 1.0)
- Example: [0.2, 0.3, 0.7, 0.8] means box from (20%, 30%) to (70%, 80%)
- This is CORNER format, NOT center format!

BOX SIZE RULES - BOXES MUST BE LARGE:
- A typical car takes 60-80% of the image
- vehicle_damage: x2-x1 should be 0.4-0.7, y2-y1 should be 0.4-0.7 (VERY LARGE BOX)
- point_of_impact: x2-x1 should be 0.3-0.5, y2-y1 should be 0.3-0.5 (LARGE BOX)
- skid_mark: x2-x1 should be 0.5-0.8 (very wide)
- debris_field: x2-x1 should be 0.3-0.6, y2-y1 should be 0.3-0.6

EXAMPLES:
- Front bumper damaged: bbox should be [0.15, 0.35, 0.75, 0.75] (covers entire front)
- Side damage: bbox should be [0.25, 0.3, 0.75, 0.85] (covers entire side)
- WRONG (too small): [0.45, 0.45, 0.55, 0.55] - TINY BOX, UNACCEPTABLE!
- CORRECT: [0.2, 0.3, 0.8, 0.8] - LARGE BOX covering damage!

NEVER make boxes smaller than 0.25 in width (x2-x1) or height (y2-y1)!

CRITICAL: Report ALL visible damage. Do not miss obvious dents, crushes, or broken parts. If you see damage, report it even if confidence is moderate (0.5+)."""


async def gemini_detect(
    image_path: str | Path,
    conf_threshold: float = 0.3,
    skip_classes: set[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Send a photo to Gemini vision and get structured evidence detections.
    Returns list of detection dicts with source="ai_inferred".
    """
    if not GEMINI_API_KEY:
        logger.warning("No Gemini API key — skipping AI detection")
        return []

    skip = skip_classes or set()
    image_path = Path(image_path)

    if not image_path.exists():
        logger.warning(f"Image not found: {image_path}")
        return []

    try:
        import httpx

        image_bytes = _resize_for_gemini(image_path, max_size=1024)
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        suffix = image_path.suffix.lower()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp"}.get(suffix, "image/jpeg")

        import asyncio
        async with httpx.AsyncClient(timeout=60) as client:
            resp = None
            max_attempts = len(GEMINI_API_KEYS) * 3
            for attempt in range(max_attempts):
                api_key = GEMINI_API_KEYS[attempt % len(GEMINI_API_KEYS)]
                resp = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                    headers={"Content-Type": "application/json"},
                    params={"key": api_key},
                    json={
                        "contents": [{"parts": [
                            {"text": DETECTION_PROMPT},
                            {"inline_data": {"mime_type": mime, "data": image_b64}},
                        ]}],
                        "generationConfig": {
                            "temperature": 0.1,
                            "maxOutputTokens": 4096,
                                                    },
                    },
                )
                if resp.status_code == 429:
                    next_key = GEMINI_API_KEYS[(attempt + 1) % len(GEMINI_API_KEYS)]
                    key_label = f"...{api_key[-6:]}" if len(api_key) > 6 else "***"
                    next_label = f"...{next_key[-6:]}" if len(next_key) > 6 else "***"
                    logger.info(f"Gemini rate limited (key {key_label}) — switching to key {next_label}")
                    await asyncio.sleep(1)
                    continue
                break

            if resp is None or resp.status_code != 200:
                logger.warning(f"Gemini vision HTTP {resp.status_code if resp else 'N/A'}")
                return []

            data = resp.json()
            candidate = data.get("candidates", [{}])[0]
            parts = candidate.get("content", {}).get("parts", [])
            text = parts[0]["text"] if parts else ""

            if not text.strip():
                logger.warning("Gemini vision returned empty response")
                return []

            detections = _parse_gemini_response(text, conf_threshold, skip)
            logger.info(f"Gemini detected {len(detections)} items in {image_path.name}")
            return detections

    except Exception as e:
        logger.warning(f"Gemini vision detection failed: {e}")
        return []


def _parse_gemini_response(text: str, conf_threshold: float, skip_classes: set[str]) -> list[dict[str, Any]]:
    """Parse Gemini's JSON response into detection dicts."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                items = json.loads(text[start:end])
            except json.JSONDecodeError:
                logger.warning(f"Could not parse Gemini response: {text[:200]}")
                return []
        else:
            logger.warning(f"No JSON array in response: {text[:200]}")
            return []

    if not isinstance(items, list):
        return []

    valid_classes = {"skid_mark", "debris_field", "point_of_impact", "vehicle_damage"}
    detections = []

    for item in items:
        if not isinstance(item, dict):
            continue
        cls = item.get("class", "")
        conf = float(item.get("confidence", 0))
        desc = item.get("description", "")
        # Support both "bbox" and "location" field names
        loc = item.get("bbox") or item.get("location") or [0.2, 0.3, 0.7, 0.7]

        if cls not in valid_classes:
            continue
        if cls in skip_classes:
            continue
        if conf < conf_threshold:
            continue
        if not isinstance(loc, list) or len(loc) != 4:
            loc = [0.5, 0.5, 0.2, 0.2]

        detections.append({
            "class": cls,
            "confidence": round(conf, 3),
            "description": desc,
            "bbox": [round(float(v), 3) for v in loc],
            "source": "ai_inferred",
        })

    return detections


# ─── Relevance Check ──────────────────────────────────────────────

RELEVANCE_PROMPT = """You are a gatekeeper for a traffic accident reconstruction system.

Look at this image and answer: Is this a photo of a traffic accident scene?

Consider it RELEVANT if it shows ANY of:
- Damaged vehicles, cars, motorcycles, trucks involved in an accident
- Road debris, broken glass, scattered vehicle parts
- Skid marks, tire marks on road surface
- Accident scene with emergency vehicles, cones, barriers
- Vehicle collision damage close-up

Consider it IRRELEVANT if it shows:
- Random objects, food, selfies, landscapes with no accident context
- Screenshots, text documents, memes
- People posing, indoor scenes, unrelated outdoor scenes

Respond with ONLY a JSON object:
{"relevant": true/false, "reason": "brief one-line explanation"}"""


async def gemini_check_relevance(image_path: str | Path) -> dict[str, Any]:
    """
    Quick check: Is this photo related to a traffic accident scene?
    Returns {"relevant": bool, "reason": str}
    Used as a fallback when YOLO + Gemini detection finds 0 evidence.
    """
    if not GEMINI_API_KEY:
        # No API key → assume relevant (don't block)
        return {"relevant": True, "reason": "No API key — skipped relevance check"}

    image_path = Path(image_path)
    if not image_path.exists():
        return {"relevant": False, "reason": "Image file not found"}

    try:
        import httpx

        with open(image_path, "rb") as f:
            image_data = f.read()
        image_b64 = base64.b64encode(image_data).decode("utf-8")

        suffix = image_path.suffix.lower()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp"}.get(suffix, "image/jpeg")

        async with httpx.AsyncClient(timeout=20) as client:
            resp = None
            for attempt in range(3):
                resp = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                    headers={"Content-Type": "application/json"},
                    params={"key": GEMINI_API_KEY},
                    json={
                        "contents": [{"parts": [
                            {"text": RELEVANCE_PROMPT},
                            {"inline_data": {"mime_type": mime, "data": image_b64}},
                        ]}],
                        "generationConfig": {
                            "temperature": 0.0,
                            "maxOutputTokens": 256,
                                                    },
                    },
                )
                if resp.status_code == 429:
                    import asyncio
                    wait = 2 ** (attempt + 1)
                    logger.info(f"Relevance check rate limited — retrying in {wait}s (attempt {attempt+1}/3)")
                    await asyncio.sleep(wait)
                    continue
                break

            if resp is None or resp.status_code != 200:
                logger.warning(f"Relevance check HTTP {resp.status_code if resp else 'N/A'}")
                return {"relevant": True, "reason": "API error — assumed relevant"}

            data = resp.json()
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            text = parts[0]["text"] if parts else ""

            return _parse_relevance_response(text)

    except Exception as e:
        logger.warning(f"Relevance check failed: {e}")
        return {"relevant": True, "reason": f"Check failed — assumed relevant"}


def _parse_relevance_response(text: str) -> dict[str, Any]:
    """Parse the relevance check JSON response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                result = json.loads(text[start:end])
            except json.JSONDecodeError:
                return {"relevant": True, "reason": "Could not parse response — assumed relevant"}
        else:
            return {"relevant": True, "reason": "No JSON in response — assumed relevant"}

    relevant = result.get("relevant", True)
    reason = result.get("reason", "")

    # Also check for string "false" in case Gemini returns it as string
    if isinstance(relevant, str):
        relevant = relevant.lower() not in ("false", "no", "0")

    return {"relevant": bool(relevant), "reason": str(reason)}


async def _openrouter_analyze_photo(
    image_path: str | Path,
    conf_threshold: float = 0.3,
) -> dict[str, Any]:
    """
    Analyze photo using OpenRouter API (Qwen vision models).
    OpenRouter uses OpenAI-compatible format.
    """
    if not OPENROUTER_API_KEY:
        logger.warning("OpenRouter analyze skipped: OPENROUTER_API_KEY is empty")
        return {"relevant": True, "reason": "No OpenRouter API key — skipped", "angle": "unknown", "detections": []}

    image_path = Path(image_path)
    if not image_path.exists():
        return {"relevant": False, "reason": "Image file not found", "detections": []}

    try:
        import httpx

        image_bytes = _resize_for_gemini(image_path, max_size=1024)
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        suffix = image_path.suffix.lower()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp"}.get(suffix, "image/jpeg")

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": COMBINED_PROMPT},
                                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                            ]
                        }
                    ],
                    "temperature": 0.0,
                    "max_tokens": 1024,
                },
            )

            if resp.status_code == 429:
                logger.warning("OpenRouter rate limited (429)")
                return {"relevant": True, "reason": "Rate limited — try again", "angle": "unknown", "detections": []}

            if resp.status_code != 200:
                logger.warning(f"OpenRouter HTTP {resp.status_code}: {resp.text[:200]}")
                return {"relevant": True, "reason": "API error", "angle": "unknown", "detections": []}

            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if not content.strip():
                logger.warning("OpenRouter returned empty response")
                return {"relevant": True, "reason": "Empty response", "angle": "unknown", "detections": []}

            result = _parse_combined_response(content, conf_threshold)
            logger.info(f"OpenRouter analyze result: relevant={result['relevant']}, angle={result['angle']}, detections={len(result['detections'])}")
            return result

    except Exception as e:
        logger.error(f"OpenRouter API error: {e}")
        return {"relevant": True, "reason": f"API error: {str(e)}", "angle": "unknown", "detections": []}


async def gemini_analyze_photo(
    image_path: str | Path,
    conf_threshold: float = 0.3,
) -> dict[str, Any]:
    """
    Single API call: returns both relevance check and detections.
    Supports both Gemini and OpenRouter (Qwen) providers.
    Returns: {"relevant": bool, "reason": str, "detections": list}
    """
    # Check which provider to use
    if VISION_PROVIDER == "openrouter" and OPENROUTER_API_KEY:
        return await _openrouter_analyze_photo(image_path, conf_threshold)
    
    # Default to Gemini
    if not GEMINI_API_KEY:
        logger.warning("Gemini analyze skipped: LLM_API_KEY is empty")
        return {"relevant": True, "reason": "No API key — skipped", "angle": "unknown", "detections": []}

    image_path = Path(image_path)
    if not image_path.exists():
        return {"relevant": False, "reason": "Image file not found", "detections": []}

    try:
        import httpx

        image_bytes = _resize_for_gemini(image_path, max_size=1024)
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        suffix = image_path.suffix.lower()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp"}.get(suffix, "image/jpeg")

        import asyncio
        async with httpx.AsyncClient(timeout=60) as client:
            resp = None
            max_attempts = max(len(GEMINI_API_KEYS) * 3, 6)
            for attempt in range(max_attempts):
                api_key = GEMINI_API_KEYS[attempt % len(GEMINI_API_KEYS)]
                try:
                    resp = await client.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                        headers={"Content-Type": "application/json"},
                        params={"key": api_key},
                        json={
                            "contents": [{"parts": [
                                {"text": COMBINED_PROMPT},
                                {"inline_data": {"mime_type": mime, "data": image_b64}},
                            ]}],
                            "generationConfig": {
                                "temperature": 0.0,
                                "maxOutputTokens": 4096,
                            },
                        },
                    )
                except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as net_err:
                    logger.warning(f"Gemini analyze network error (attempt {attempt+1}): {net_err}")
                    await asyncio.sleep(2 ** min(attempt, 3))
                    continue

                if resp.status_code == 429:
                    retry_seconds = 2
                    try:
                        err_json = resp.json()
                        msg = err_json.get("error", {}).get("message", "")
                        import re
                        m = re.search(r"retry in ([0-9]+\.?[0-9]*)", msg, re.IGNORECASE)
                        if m:
                            retry_seconds = max(1, int(float(m.group(1))))
                    except Exception:
                        pass
                    next_key = GEMINI_API_KEYS[(attempt + 1) % len(GEMINI_API_KEYS)]
                    key_label = f"...{api_key[-6:]}" if len(api_key) > 6 else "***"
                    next_label = f"...{next_key[-6:]}" if len(next_key) > 6 else "***"
                    logger.info(f"Gemini analyze rate limited (key {key_label}) — switching to key {next_label}, retry in {retry_seconds}s")
                    await asyncio.sleep(retry_seconds)
                    continue

                if resp.status_code >= 500:
                    logger.warning(f"Gemini analyze server error {resp.status_code}, retrying...")
                    await asyncio.sleep(2 ** min(attempt, 3))
                    continue

                break

            if resp is None or resp.status_code != 200:
                error_text = resp.text[:500] if resp else "N/A"
                logger.warning(f"Gemini analyze HTTP {resp.status_code if resp else 'N/A'}: {error_text}")
                return {"relevant": True, "reason": "API error — assumed relevant", "angle": "unknown", "detections": []}

            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                logger.warning(f"Gemini analyze returned no candidates: {resp.text[:500]}")
                return {"relevant": True, "reason": "No AI response candidates", "angle": "unknown", "detections": []}
            candidate = candidates[0]
            if candidate.get("finishReason") not in (None, "STOP"):
                logger.warning(f"Gemini analyze finishReason={candidate.get('finishReason')}")
            parts = candidate.get("content", {}).get("parts", [])
            text = parts[0]["text"] if parts else ""

            result = _parse_combined_response(text, conf_threshold)
            logger.info(f"Gemini analyze result: relevant={result['relevant']}, angle={result['angle']}, detections={len(result['detections'])}")
            return result

    except Exception as e:
        logger.warning(f"Gemini analyze failed: {e}")
        return {"relevant": True, "reason": f"Analysis failed — assumed relevant", "detections": []}


def _parse_combined_response(text: str, conf_threshold: float) -> dict[str, Any]:
    """Parse the combined relevance + detections response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                result = json.loads(text[start:end])
            except json.JSONDecodeError:
                return {"relevant": True, "reason": "Could not parse response", "angle": "unknown", "detections": []}
        else:
            return {"relevant": True, "reason": "No JSON in response", "angle": "unknown", "detections": []}

    relevant = result.get("relevant", True)
    if isinstance(relevant, str):
        relevant = relevant.lower() not in ("false", "no", "0")

    detections = []
    valid_classes = {"skid_mark", "debris_field", "point_of_impact", "vehicle_damage"}
    for item in result.get("detections", []):
        if not isinstance(item, dict):
            continue
        cls = item.get("class", "")
        conf = float(item.get("confidence", 0))
        desc = item.get("description", "")
        # Support both "bbox" and "location" field names
        loc = item.get("bbox") or item.get("location") or [0.2, 0.3, 0.7, 0.7]

        if cls not in valid_classes:
            continue
        if conf < conf_threshold:
            continue
        if not isinstance(loc, list) or len(loc) != 4:
            loc = [0.2, 0.3, 0.5, 0.6]
        
        # Normalize bbox values to 0-1 range (model might return percentages 0-100)
        loc = [float(v) for v in loc]
        if any(v > 1.0 for v in loc):
            loc = [v / 100.0 for v in loc]
        
        # Handle both [x1, y1, x2, y2] and [x_center, y_center, width, height] formats
        # If x2 < x1 or y2 < y1, it's likely [x_center, y_center, width, height]
        x1, y1, x2, y2 = loc
        if x2 < x1 or y2 < y1:
            # Convert from [x_center, y_center, width, height] to [x1, y1, x2, y2]
            xc, yc, w, h = loc
            x1 = max(0.0, xc - w/2)
            y1 = max(0.0, yc - h/2)
            x2 = min(1.0, xc + w/2)
            y2 = min(1.0, yc + h/2)
        else:
            # Already [x1, y1, x2, y2], just clamp
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))
        
        # Ensure x1 < x2 and y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        detections.append({
            "class": cls,
            "confidence": round(conf, 3),
            "description": desc,
            "bbox": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
            "source": "ai_inferred",
        })

    angle = result.get("angle", "unknown")
    if angle not in ("front", "left", "right", "rear"):
        angle = "unknown"

    return {
        "relevant": bool(relevant),
        "reason": result.get("reason", ""),
        "angle": angle,
        "detections": detections,
    }

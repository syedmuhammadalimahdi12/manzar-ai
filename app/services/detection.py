"""
MANZAR — YOLOv8 Evidence Detection Service
============================================
Runs the fine-tuned YOLOv8s model on uploaded photos to detect:
  - skid_mark
  - vehicle_damage
  - debris_field
  - point_of_impact

Model: models/FINAL_best_model.pt (YOLOv8s, 80 epochs, imgsz=960)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("manzar.detection")

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "FINAL_best_model.pt"

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        logger.error(f"Model not found: {MODEL_PATH}")
        return None
    try:
        from ultralytics import YOLO
        _model = YOLO(str(MODEL_PATH))
        logger.info(f"YOLOv8 model loaded: {MODEL_PATH}")
        logger.info(f"Classes: {_model.names}")
        return _model
    except ImportError:
        logger.warning("ultralytics not installed — detection disabled")
        return None
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


def detect_evidence(image_path: str | Path, conf_threshold: float = 0.25) -> list[dict[str, Any]]:
    """Run YOLOv8 detection on a single image. Returns list of detections."""
    model = _load_model()
    if model is None:
        return []
    image_path = Path(image_path)
    if not image_path.exists():
        logger.warning(f"Image not found: {image_path}")
        return []
    try:
        results = model(str(image_path), conf=conf_threshold, verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                bbox = boxes.xyxy[i].tolist()
                detections.append({
                    "class": model.names.get(cls_id, f"class_{cls_id}"),
                    "class_id": cls_id,
                    "confidence": round(conf, 4),
                    "bbox": [round(v, 1) for v in bbox],
                })
        logger.info(f"Detection on {image_path.name}: {len(detections)} objects")
        return detections
    except Exception as e:
        logger.error(f"Detection failed on {image_path}: {e}")
        return []


def detect_all_photos(photo_records: list[dict], conf_threshold: float = 0.25) -> dict[str, list[dict]]:
    """Run detection on all photos. Returns {filename: [detections...]}."""
    results = {}
    for photo in photo_records:
        path = photo.get("path", "")
        filename = photo.get("filename", "")
        if path and Path(path).exists():
            results[filename] = detect_evidence(path, conf_threshold)
    return results


def summarize_detections(all_detections: dict[str, list[dict]]) -> dict[str, Any]:
    """Summarize detection results across all photos."""
    total = 0
    by_class: dict[str, int] = {}
    photos_with = 0
    for filename, dets in all_detections.items():
        if dets:
            photos_with += 1
        total += len(dets)
        for d in dets:
            cls = d["class"]
            by_class[cls] = by_class.get(cls, 0) + 1
    return {
        "total_detections": total,
        "by_class": by_class,
        "photos_with_detections": photos_with,
        "photos_without_detections": len(all_detections) - photos_with,
    }

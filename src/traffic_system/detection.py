"""
detection.py — YOLOv8 model loading and per-frame vehicle detection.

Pure detection concerns only: load the model, run it on a frame, apply
per-class confidence and size filters, and return a plain list of
(bbox, confidence, class_name) tuples. ROI filtering, tracking, and signal
logic live in pipeline.py / tracking.py / signal_logic.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from . import config

logger = logging.getLogger(__name__)

_model_cache: dict[str, "object"] = {}


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def centre(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def as_deepsort_input(self) -> tuple[list[int], float, int]:
        """Format expected by DeepSort's update_tracks(): ([x, y, w, h], conf, class_id)."""
        return [self.x1, self.y1, self.width, self.height], self.confidence, self.class_id


def get_model(model_path: str = config.DEFAULT_MODEL_PATH):
    """Load (and cache) a YOLO model by path. Cached per path so switching
    models in the dashboard doesn't require a process restart."""
    if model_path not in _model_cache:
        from ultralytics import YOLO  # imported lazily so tests can run without it

        logger.info("Loading YOLO model: %s", model_path)
        _model_cache[model_path] = YOLO(model_path)
    return _model_cache[model_path]


def detect_vehicles(
    frame, model, frame_w: int, frame_h: int,
    min_conf_override: float | None = None,
) -> list[Detection]:
    """Run YOLO on a frame and return filtered vehicle detections.

    Filters applied, in order:
      1. class must be in config.VEHICLE_CLASSES
      2. confidence must clear the per-class threshold (config.CONF_PER_CLASS),
         raised to `min_conf_override` if that's higher (never lowered below
         the tuned per-class default — the dashboard slider is a ceiling
         raise, not a full override, so it can't accidentally let in more
         noise than the tuned thresholds allow)
      3. box must be at least MIN_BOX_W x MIN_BOX_H (rejects noise)
      4. box must not exceed MAX_BOX_FRAC_OF_FRAME of the frame (rejects
         ghost/degenerate boxes)
    """
    results = model(
        frame,
        conf=config.YOLO_CONF,
        iou=config.YOLO_IOU,
        imgsz=config.YOLO_IMGSZ,
        verbose=False,
    )[0]

    detections: list[Detection] = []
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls not in config.VEHICLE_CLASSES:
            continue

        conf_score = float(box.conf[0])
        threshold = config.CONF_PER_CLASS.get(cls, config.DEFAULT_MIN_CONF)
        if min_conf_override is not None:
            threshold = max(threshold, min_conf_override)
        if conf_score < threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_w, box_h = x2 - x1, y2 - y1

        if box_w < config.MIN_BOX_W or box_h < config.MIN_BOX_H:
            continue
        if box_w > frame_w * config.MAX_BOX_FRAC_OF_FRAME or box_h > frame_h * config.MAX_BOX_FRAC_OF_FRAME:
            continue

        detections.append(
            Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=conf_score,
                class_id=cls,
                class_name=config.CLASS_NAMES.get(cls, model.names[cls]),
            )
        )

    return detections

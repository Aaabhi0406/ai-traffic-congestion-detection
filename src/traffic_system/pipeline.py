"""
pipeline.py — orchestrates detection -> ROI filter -> tracking -> line
crossing -> signal logic -> ML prediction for a single video frame.

This is the single source of truth used by both cli.py (OpenCV window mode)
and dashboard.py (Streamlit). Previously main.py duplicated this logic with
its own (slightly different, and buggier — see CHANGELOG/README) copy.

ARCHITECTURE — how every number flows:
═══════════════════════════════════════════════════════════════════════════

  VIDEO FRAME
       |
       v
  STEP 1 — ROI (Region of Interest)
    A trapezoid polygon covering the visible road. Only vehicles whose
    bounding-box centre falls INSIDE the ROI are counted. Vehicles outside
    it are detected but ignored (drawn in grey, not counted).
       |
       v
  STEP 2 — YOLOv8 detection (detection.py)
    Per-class confidence thresholds (config.CONF_PER_CLASS), min/max box
    size filters. Runs at imgsz=1280 so small vehicles (motorcycles,
    bicycles) are still resolvable.
       |
       v
  STEP 3 — ROI filter
    For each detection: is the box centre inside the ROI polygon?
      YES -> counted, green box
      NO  -> not counted, grey box
       |
       v
  STEP 4 — DeepSort tracking (tracking.py), ROI detections only
    Assigns a persistent ID to each ROI vehicle (orange box + #ID). Ghost
    tracks with no nearby real detection are suppressed. Tracking IDs are
    NOT used for counting -- only for line-crossing and visual display.
       |
       v
  STEP 5 — Counting line (tracking.LineCrossingCounter)
    A horizontal line across the ROI. When a tracked vehicle's centre
    crosses it top -> bottom, a crossing event is recorded. Used as a
    throughput feature, not for the per-frame vehicle_count.
       |
       v
  STEP 6 — Signal logic (signal_logic.get_signal_time)
    vehicle_count (ROI, this frame) -> traffic level -> green_time.
       |
       v
  STEP 7 — Random Forest prediction (signal_logic.TrafficDensityPredictor)
    Predicts weighted density config.RF_PREDICT_HORIZON frames ahead from a
    config.RF_WINDOW_SIZE-frame rolling window.
       |
       v
  STEP 8 — Result dict
    Every number the dashboard/CLI displays comes from this one dict.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import config
from .detection import Detection, detect_vehicles, get_model
from .signal_logic import TrafficDensityPredictor, get_signal_time
from .tracking import LineCrossingCounter, build_tracker, has_nearby_detection

__all__ = ["TrafficPipeline", "build_roi", "centre_in_roi", "get_model"]


def build_roi(frame_h: int, frame_w: int) -> np.ndarray:
    """Trapezoid ROI covering the full visible road, with a slight top
    inset for perspective (the far lane appears narrower than the near
    one). Adjust the config.ROI_* fractions if your camera angle differs."""
    top_y = int(frame_h * config.ROI_TOP_Y_FRAC)
    top_l_x = int(frame_w * config.ROI_TOP_LEFT_X_FRAC)
    top_r_x = int(frame_w * config.ROI_TOP_RIGHT_X_FRAC)
    bot_y = int(frame_h * config.ROI_BOTTOM_Y_FRAC)

    return np.array(
        [
            [top_l_x, top_y],
            [top_r_x, top_y],
            [frame_w, bot_y],
            [0, bot_y],
        ],
        dtype=np.int32,
    )


def centre_in_roi(cx: float, cy: float, roi: np.ndarray) -> bool:
    """True if point (cx, cy) is inside the ROI polygon."""
    return cv2.pointPolygonTest(roi, (float(cx), float(cy)), False) >= 0


class TrafficPipeline:
    """Stateful per-camera pipeline: owns the tracker, the line-crossing
    counter, and the density predictor. Create one instance per video
    source; call process_frame() once per frame.
    """

    def __init__(self, density_predictor: TrafficDensityPredictor | None = None):
        self.tracker = build_tracker()
        self.crossing_counter = LineCrossingCounter()
        self.density_predictor = density_predictor or TrafficDensityPredictor()

    def reset(self) -> None:
        """Reset all per-video state (call when switching video sources)."""
        self.tracker = build_tracker()
        self.crossing_counter.reset()
        self.density_predictor = TrafficDensityPredictor()

    def process_frame(self, frame: np.ndarray, model, min_conf_override: float | None = None) -> dict:
        h, w = frame.shape[:2]
        roi = build_roi(h, w)
        line_y = int(h * config.COUNTING_LINE_Y_FRAC)

        self._draw_roi_and_line(frame, roi, line_y)

        detections = detect_vehicles(frame, model, frame_w=w, frame_h=h, min_conf_override=min_conf_override)

        roi_detections: list[Detection] = []
        vehicle_types: dict[str, int] = {}
        total_weight = 0.0
        vehicle_count = 0

        for det in detections:
            cx, cy = det.centre
            in_roi = centre_in_roi(cx, cy, roi)

            if in_roi:
                vehicle_count += 1
                vehicle_types[det.class_name] = vehicle_types.get(det.class_name, 0) + 1
                total_weight += config.VEHICLE_WEIGHTS.get(det.class_name, 1.0)
                roi_detections.append(det)
                self._draw_box(frame, det, config.COLORS.roi_in, label=f"{det.class_name} {det.confidence:.2f}")
            else:
                self._draw_box(frame, det, config.COLORS.roi_out, label=det.class_name, thickness=1)

        crossing_total = self._track_and_count_crossings(frame, roi_detections, line_y)

        green_time, traffic_level = get_signal_time(vehicle_count)

        self.density_predictor.update(total_weight)
        pred_density, pred_level, pred_conf = self.density_predictor.predict()

        self._draw_hud(frame, vehicle_count, crossing_total, traffic_level, green_time, pred_level, pred_conf)

        return {
            "annotated_frame": frame,
            "vehicle_count": vehicle_count,
            "vehicle_types": vehicle_types,
            "total_weight": total_weight,
            "green_time": green_time,
            "traffic_level": traffic_level,
            "crossing_count": crossing_total,
            "pred_density": pred_density,
            "pred_level": pred_level,
            "pred_confidence": pred_conf,
        }

    # ── drawing helpers ──────────────────────────────────────────────────────

    def _draw_roi_and_line(self, frame: np.ndarray, roi: np.ndarray, line_y: int) -> None:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.fillPoly(overlay, [roi], (0, 255, 100))
        cv2.addWeighted(overlay, 0.07, frame, 0.93, 0, frame)
        cv2.polylines(frame, [roi], True, (0, 255, 100), 2, cv2.LINE_AA)

        cv2.line(frame, (0, line_y), (w, line_y), config.COLORS.line, 2, cv2.LINE_AA)
        cv2.putText(
            frame, "COUNTING LINE", (8, line_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COLORS.line, 1, cv2.LINE_AA,
        )

    def _draw_box(self, frame: np.ndarray, det: Detection, color, label: str, thickness: int = 2) -> None:
        cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, thickness)
        cv2.putText(
            frame, label, (det.x1, max(det.y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA,
        )

    def _track_and_count_crossings(self, frame: np.ndarray, roi_detections: list[Detection], line_y: int) -> int:
        h, w = frame.shape[:2]
        deepsort_input = [d.as_deepsort_input() for d in roi_detections]
        detection_centres = [d.centre for d in roi_detections]

        tracks = self.tracker.update_tracks(deepsort_input, frame=frame)
        curr_centres: dict[int, float] = {}

        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            left, top, right, bottom = map(int, track.to_ltrb())
            left, top = max(0, left), max(0, top)
            right, bottom = min(w, right), min(h, bottom)

            if not has_nearby_detection(left, top, right, bottom, detection_centres):
                continue

            cy_track = (top + bottom) / 2.0
            curr_centres[tid] = cy_track

            cv2.rectangle(frame, (left, top), (right, bottom), config.COLORS.track, 2)
            cv2.putText(
                frame, f"#{tid}", (left, max(top - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, config.COLORS.track, 1, cv2.LINE_AA,
            )

            crossed = self.crossing_counter.update(tid, cy_track, line_y)
            if crossed:
                cv2.line(frame, (0, line_y), (w, line_y), config.COLORS.line_flash, 3, cv2.LINE_AA)

        self.crossing_counter.commit_frame(curr_centres)
        return self.crossing_counter.total

    def _draw_hud(
        self, frame: np.ndarray, vehicle_count: int, crossing_total: int,
        traffic_level: str, green_time: int, pred_level: str, pred_conf: float,
    ) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (320, 170), config.COLORS.hud_bg, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        hud = [
            (f"IN ROI      : {vehicle_count}", (10, 26), (0, 220, 255)),
            (f"CROSSED LINE: {crossing_total}", (10, 52), config.COLORS.line),
            (f"TRAFFIC     : {traffic_level}", (10, 78), (255, 220, 0)),
            (f"GREEN TIME  : {green_time}s", (10, 104), (0, 255, 120)),
            (f"PREDICTED   : {pred_level}", (10, 130), (200, 100, 255)),
            (f"RF CONF     : {pred_conf:.0%}", (10, 156), (160, 160, 160)),
        ]
        for text, pos, color in hud:
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)

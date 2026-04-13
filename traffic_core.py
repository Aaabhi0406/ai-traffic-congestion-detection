"""
traffic_core.py — Precise detection with per-class confidence + ROI line of sight

ARCHITECTURE — how every number flows:
═══════════════════════════════════════════════════════════════════════════════

  VIDEO FRAME
       │
       ▼
  ┌─────────────────────────────────────────────────────────┐
  │  STEP 1 — ROI (Region of Interest)                      │
  │  A polygon is drawn on the frame. Only vehicles whose   │
  │  bounding box centre falls INSIDE the ROI are counted.  │
  │  Vehicles outside ROI are detected but ignored.         │
  └───────────────────────┬─────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────┐
  │  STEP 2 — YOLOv8 Detection (per-class confidence)       │
  │  car/bus/truck  → conf ≥ 0.45                           │
  │  motorcycle     → conf ≥ 0.22  (small, needs low conf)  │
  │  bicycle        → conf ≥ 0.22  (very small object)      │
  │  iou = 0.4      → removes duplicate boxes               │
  │                                                         │
  │  OUTPUT: list of boxes, each labelled with class name   │
  └───────────────────────┬─────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────┐
  │  STEP 3 — ROI Filter                                    │
  │  For each detected box: is box centre inside ROI?       │
  │    YES → count it, draw GREEN box                       │
  │    NO  → draw GREY box (visible but not counted)        │
  └───────────────────────┬─────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────┐
  │  STEP 4 — DeepSort Tracking (inside ROI only)           │
  │  Assigns persistent ID to each ROI vehicle.             │
  │  Draws ORANGE box + #ID label.                          │
  │  NOT used for counting — only for visual ID display.    │
  └───────────────────────┬─────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────┐
  │  STEP 5 — Counting Line (Line of Sight)                 │
  │  A horizontal line across the ROI.                      │
  │  When a tracked vehicle's centre crosses this line      │
  │  (top→bottom), a CROSSING EVENT is recorded.            │
  │  crossing_count = total vehicles that crossed line      │
  │  (used in prediction as throughput feature)             │
  └───────────────────────┬─────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────┐
  │  STEP 6 — Signal Logic (signal_logic.py)                │
  │  vehicle_count (inside ROI this frame) → traffic level  │
  │  LOW:    0–6  vehicles → 15s green                      │
  │  MEDIUM: 7–12 vehicles → 25s green                      │
  │  HIGH:   13+  vehicles → 40s green                      │
  └───────────────────────┬─────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────┐
  │  STEP 7 — Random Forest Prediction (signal_logic.py)    │
  │  Features: rolling 30-frame window of weighted density  │
  │  Predicts density 15 frames (~0.5s) ahead               │
  │  Output: pred_density, pred_level, confidence (R²)      │
  └───────────────────────┬─────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────┐
  │  STEP 8 — Dashboard output dict                         │
  │  ALL numbers on dashboard come from this one dict.      │
  │  vehicle_count  = ROI detections this frame             │
  │  vehicle_types  = same detections broken down by class  │
  │  traffic_level  = from vehicle_count                    │
  │  green_time     = from traffic_level                    │
  │  crossing_count = line-crossing throughput              │
  └─────────────────────────────────────────────────────────┘

"""

import cv2
import numpy as np
from ultralytics import YOLO
from tracker import tracker
from signal_logic import get_signal_time, density_predictor

# ── COCO class IDs ────────────────────────────────────────────────────────────
VEHICLE_CLASSES = [1, 2, 3, 5, 7]   # bicycle, car, motorcycle, bus, truck

# Per-class confidence thresholds
# At imgsz=1280 YOLO sees small objects better, but motorcycles in dense
# Indian traffic (far lane, partially occluded) still score lower than cars.
# These are the minimum confidence values per class:
CONF_PER_CLASS = {
    1: 0.30,   # bicycle
    2: 0.45,   # car
    3: 0.30,   # motorcycle  ← still lower than car but not so low pedestrians sneak in
    5: 0.45,   # bus
    7: 0.45,   # truck
}

# Minimum box dimensions in pixels — rejects tiny noise detections
# A real motorcycle at road level is at least 20x20px even when far
MIN_BOX_W = 20
MIN_BOX_H = 20

# Weights for density score (fed to ML predictor only, NOT used for counting)
VEHICLE_WEIGHTS = {
    "bicycle":    0.3,
    "car":        1.0,
    "motorcycle": 0.5,
    "bus":        2.0,
    "truck":      2.0,
}

# Box colours
COL_ROI_IN  = (0, 220, 80)    # green  — inside ROI, counted
COL_ROI_OUT = (120, 120, 120) # grey   — outside ROI, not counted
COL_TRACK   = (255, 140, 0)   # orange — DeepSort ID label
COL_LINE    = (0, 200, 255)   # cyan   — counting line
COL_HUD     = (0, 0, 0)       # HUD background

_model = None

# ── Line-crossing state (persists across frames) ──────────────────────────────
# Maps track_id → last known y-centre
_prev_centres: dict[int, float] = {}
crossing_count: int = 0


def get_model(model_path: str = "yolov8s.pt") -> YOLO:  # s=small, much better for motorcycles
    global _model
    if _model is None:
        _model = YOLO(model_path)
    return _model


def _build_roi(frame_h: int, frame_w: int) -> np.ndarray:
    """
    ROI = the FULL lane visible in the CCTV frame.

    The camera sees a wide road from an overhead/angled CCTV view.
    We cover the entire visible road from top-edge to bottom-edge,
    with a slight perspective taper (road appears narrower far away).

    Visual on a 1920x1080 frame:
        (48, 2)  ─────────────────── (1872, 2)    ← top: nearly full width
           |                                |
           |       FULL VISIBLE ROAD        |
           |                                |
        (0, 1078) ──────────────── (1920, 1078)   ← bottom: full width

    If your camera angle cuts the road differently, adjust top_l/top_r.
    """
    # Top edge: slight inset for perspective (far-away lane is narrower)
    top_y   = int(frame_h * 0.02)   # almost top of frame
    top_l_x = int(frame_w * 0.025)  # 2.5% from left at top
    top_r_x = int(frame_w * 0.975)  # 2.5% from right at top

    # Bottom edge: full width
    bot_y   = int(frame_h * 0.99)
    bot_l_x = 0
    bot_r_x = frame_w

    return np.array([
        [top_l_x, top_y],   # top-left  (near top of frame)
        [top_r_x, top_y],   # top-right (near top of frame)
        [bot_r_x, bot_y],   # bottom-right (full width)
        [bot_l_x, bot_y],   # bottom-left  (full width)
    ], dtype=np.int32)


def _centre_in_roi(cx: float, cy: float, roi: np.ndarray) -> bool:
    """Return True if point (cx, cy) is inside the ROI polygon."""
    return cv2.pointPolygonTest(roi, (float(cx), float(cy)), False) >= 0


def process_frame(frame: np.ndarray, model: YOLO) -> dict:
    global crossing_count, _prev_centres

    h, w = frame.shape[:2]
    roi  = _build_roi(h, w)

    # Counting line: horizontal line at 60% frame height inside ROI
    line_y = int(h * 0.75)   # 75% down — lower half of full-frame ROI

    # ── Step 1: Draw ROI polygon ──────────────────────────────────────────────
    roi_overlay = frame.copy()
    cv2.fillPoly(roi_overlay, [roi], (0, 255, 100))
    cv2.addWeighted(roi_overlay, 0.07, frame, 0.93, 0, frame)
    cv2.polylines(frame, [roi], True, (0, 255, 100), 2, cv2.LINE_AA)

    # ── Step 2: Draw counting line ────────────────────────────────────────────
    cv2.line(frame, (0, line_y), (w, line_y), COL_LINE, 2, cv2.LINE_AA)
    cv2.putText(frame, "COUNTING LINE", (8, line_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_LINE, 1, cv2.LINE_AA)

    # ── Step 3: YOLO detection ────────────────────────────────────────────────
    # imgsz=1280  → higher resolution so motorcycles appear bigger in input
    # conf=0.30   → balanced: catches motorcycles but blocks pedestrian noise
    # iou=0.45    → suppress duplicate boxes on same vehicle
    results = model(frame, conf=0.30, iou=0.45, imgsz=1280, verbose=False)[0]

    detections_roi = []   # DeepSort input — only inside ROI
    vehicle_types: dict[str, int] = {}
    total_weight  = 0.0
    vehicle_count = 0     # authoritative count: ROI detections this frame

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls not in VEHICLE_CLASSES:
            continue

        conf_score = float(box.conf[0])
        # Apply per-class threshold
        if conf_score < CONF_PER_CLASS.get(cls, 0.45):
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_w, box_h = x2 - x1, y2 - y1

        # Reject boxes too small to be real vehicles
        if box_w < MIN_BOX_W or box_h < MIN_BOX_H:
            continue

        # Reject boxes that are too large to be a single vehicle
        # (DeepSort ghost tracks sometimes produce full-frame boxes)
        if box_w > w * 0.85 or box_h > h * 0.85:
            continue

        name = model.names[cls]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        in_roi = _centre_in_roi(cx, cy, roi)

        if in_roi:
            # ── Counted vehicle ──
            vehicle_count += 1
            vehicle_types[name] = vehicle_types.get(name, 0) + 1
            total_weight += VEHICLE_WEIGHTS.get(name, 1.0)
            detections_roi.append(([x1, y1, x2 - x1, y2 - y1], conf_score, cls))

            # Green box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), COL_ROI_IN, 2)
            cv2.putText(frame, f"{name} {conf_score:.2f}",
                        (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, COL_ROI_IN, 1, cv2.LINE_AA)
        else:
            # Grey box — detected but outside ROI, not counted
            cv2.rectangle(frame, (x1, y1), (x2, y2), COL_ROI_OUT, 1)
            cv2.putText(frame, f"{name}",
                        (x1, max(y1 - 4, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_ROI_OUT, 1, cv2.LINE_AA)

    # ── Step 4: DeepSort tracking (ROI vehicles only) ─────────────────────────
    # Build set of YOLO detection centres for this frame
    # DeepSort orange box is ONLY drawn if it overlaps a real YOLO detection
    # This prevents ghost boxes being drawn when no vehicle is actually there
    yolo_centres = [(
        det[0][0] + det[0][2] / 2,   # cx = x + w/2
        det[0][1] + det[0][3] / 2    # cy = y + h/2
    ) for det in detections_roi]

    def _has_nearby_detection(lx, ty, rx, by, centres, threshold=60):
        """Return True if any YOLO detection centre falls inside or near this track box."""
        for cx, cy in centres:
            if lx - threshold < cx < rx + threshold and ty - threshold < cy < by + threshold:
                return True
        return False

    tracks = tracker.update_tracks(detections_roi, frame=frame)
    curr_centres: dict[int, float] = {}

    for track in tracks:
        if not track.is_confirmed():
            continue
        tid = track.track_id
        l, t, r, b = map(int, track.to_ltrb())

        # Clamp to frame bounds
        l, t = max(0, l), max(0, t)
        r, b = min(w, r), min(h, b)

        # Skip ghost tracks — only draw if a real YOLO detection is nearby
        if not _has_nearby_detection(l, t, r, b, yolo_centres):
            continue

        cy_track = (t + b) / 2.0
        curr_centres[tid] = cy_track

        # Orange tracking box + ID
        cv2.rectangle(frame, (l, t), (r, b), COL_TRACK, 2)
        cv2.putText(frame, f"#{tid}", (l, max(t - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, COL_TRACK, 1, cv2.LINE_AA)

        # ── Step 5: Line crossing detection ──────────────────────────────────
        if tid in _prev_centres:
            prev_cy = _prev_centres[tid]
            # Vehicle moved from above line to below line (top→bottom crossing)
            if prev_cy < line_y <= cy_track:
                crossing_count += 1
                # Flash the line yellow on crossing
                cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 3, cv2.LINE_AA)

    _prev_centres = curr_centres

    # ── Step 6: Signal logic ──────────────────────────────────────────────────
    green_time, traffic_level = get_signal_time(vehicle_count)

    # ── Step 7: ML density prediction ────────────────────────────────────────
    density_predictor.update(total_weight)
    pred_density, pred_level, pred_conf = density_predictor.predict()

    # ── Step 8: HUD overlay ───────────────────────────────────────────────────
    hud_overlay = frame.copy()
    cv2.rectangle(hud_overlay, (0, 0), (320, 170), (0, 0, 0), -1)
    cv2.addWeighted(hud_overlay, 0.65, frame, 0.35, 0, frame)

    hud = [
        (f"IN ROI      : {vehicle_count}",    (10, 26),  (0, 220, 255)),
        (f"CROSSED LINE: {crossing_count}",   (10, 52),  COL_LINE),
        (f"TRAFFIC     : {traffic_level}",    (10, 78),  (255, 220, 0)),
        (f"GREEN TIME  : {green_time}s",       (10, 104), (0, 255, 120)),
        (f"PREDICTED   : {pred_level}",        (10, 130), (200, 100, 255)),
        (f"RF CONF     : {pred_conf:.0%}",     (10, 156), (160, 160, 160)),
    ]
    for text, pos, color in hud:
        cv2.putText(frame, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)

    return {
        "annotated_frame":  frame,
        "vehicle_count":    vehicle_count,
        "vehicle_types":    vehicle_types,
        "total_weight":     total_weight,
        "green_time":       green_time,
        "traffic_level":    traffic_level,
        "crossing_count":   crossing_count,
        "pred_density":     pred_density,
        "pred_level":       pred_level,
        "pred_confidence":  pred_conf,
    }

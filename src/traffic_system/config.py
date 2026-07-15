"""
config.py — every tunable constant in one place.

Previously these were scattered (and in main.py's case, duplicated with
slightly different values) across main.py and traffic_core.py. Reconciled
here to the traffic_core.py values, which were the more recently tuned set.
"""

from __future__ import annotations

from dataclasses import dataclass

# ── COCO class IDs we care about ─────────────────────────────────────────────
# 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES: list[int] = [1, 2, 3, 5, 7]

CLASS_NAMES: dict[int, str] = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Per-class minimum confidence.
# At imgsz=1280 YOLO sees small objects better, but motorcycles in dense
# traffic (far lane, partially occluded) still score lower than cars, so
# they get a lower bar. Values below were tuned against a decent chunk of
# manual review, not defaults.
CONF_PER_CLASS: dict[int, float] = {
    1: 0.30,  # bicycle
    2: 0.45,  # car
    3: 0.30,  # motorcycle
    5: 0.45,  # bus
    7: 0.45,  # truck
}
DEFAULT_MIN_CONF = 0.45

# Reject boxes too small to plausibly be a vehicle (noise) or too large to be
# a single vehicle (DeepSort ghost tracks sometimes produce near-full-frame
# boxes).
MIN_BOX_W = 20
MIN_BOX_H = 20
MAX_BOX_FRAC_OF_FRAME = 0.85

# Detection call parameters
YOLO_IMGSZ = 1280
YOLO_CONF = 0.30   # coarse pre-filter; CONF_PER_CLASS applies the real cutoff
YOLO_IOU = 0.45

DEFAULT_MODEL_PATH = "yolov8s.pt"  # 's' = small: notably better motorcycle
                                    # recall than 'n' in our test clips

# Weights for the density score fed to the ML predictor only.
# NOT used for the raw vehicle_count used in signal timing.
VEHICLE_WEIGHTS: dict[str, float] = {
    "bicycle": 0.3,
    "car": 1.0,
    "motorcycle": 0.5,
    "bus": 2.0,
    "truck": 2.0,
}

# ── ROI (region of interest) geometry, as fractions of frame size ───────────
ROI_TOP_Y_FRAC = 0.02
ROI_TOP_LEFT_X_FRAC = 0.025
ROI_TOP_RIGHT_X_FRAC = 0.975
ROI_BOTTOM_Y_FRAC = 0.99

# Counting line position, as a fraction of frame height
COUNTING_LINE_Y_FRAC = 0.75

# ── DeepSort tracker parameters ──────────────────────────────────────────────
# Tuned from defaults after visible ghost-track issues at higher max_age /
# looser cosine distance:
#   max_age 20 -> 5           (ghost boxes lingered and expanded across frame)
#   n_init 1 -> 2             (a single noisy detection became a confirmed track)
#   nms_max_overlap 0.7 -> 0.5
#   max_cosine_distance 0.4 -> 0.25  (too loose -> identity switches)
DEEPSORT_MAX_AGE = 5
DEEPSORT_N_INIT = 2
DEEPSORT_NMS_MAX_OVERLAP = 0.5
DEEPSORT_MAX_COSINE_DISTANCE = 0.25
DEEPSORT_NN_BUDGET = 100

# ── Signal timing bands ──────────────────────────────────────────────────────
# Based on vehicles counted INSIDE the ROI in a single frame.
#   LOW    : 0  - LOW_MAX       -> 15s green
#   MEDIUM : LOW_MAX+1 - MEDIUM_MAX -> 25s green
#   HIGH   : > MEDIUM_MAX       -> 40s green
LOW_MAX = 12
MEDIUM_MAX = 25
GREEN_TIME: dict[str, int] = {"LOW": 15, "MEDIUM": 25, "HIGH": 40}

# ── Density thresholds for the ML predictor's level labels ─────────────────
# Separate scale from the raw-count bands above, because this operates on
# *weighted* density (see VEHICLE_WEIGHTS), not a raw vehicle count.
DENSITY_LOW = 6.0
DENSITY_MEDIUM = 14.0

# ── Random Forest density predictor ─────────────────────────────────────────
RF_WINDOW_SIZE = 30
RF_PREDICT_HORIZON = 15
RF_MIN_TRAIN_SAMPLES = 80
RF_RETRAIN_EVERY = 30
RF_MAX_TRAIN_HISTORY = 500
RF_N_ESTIMATORS = 50
RF_MAX_DEPTH = 6
RF_RANDOM_STATE = 42

# ── Drawing colours (BGR, OpenCV convention) ────────────────────────────────
@dataclass(frozen=True)
class Colors:
    roi_in: tuple[int, int, int] = (0, 220, 80)      # green — inside ROI, counted
    roi_out: tuple[int, int, int] = (120, 120, 120)  # grey — outside ROI
    track: tuple[int, int, int] = (255, 140, 0)      # orange — DeepSort ID
    line: tuple[int, int, int] = (0, 200, 255)       # cyan — counting line
    line_flash: tuple[int, int, int] = (0, 255, 255) # yellow — on crossing
    hud_bg: tuple[int, int, int] = (0, 0, 0)


COLORS = Colors()

KNOWN_VIDEOS: dict[str, str] = {
    "14552311-hd_1920_1080_50fps.mp4": "Video 1 · 50fps · 1080p",
    "15300538-hd_1920_1080_60fps.mp4": "Video 2 · 60fps · 1080p",
}

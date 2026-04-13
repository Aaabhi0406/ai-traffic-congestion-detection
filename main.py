"""
main.py — OpenCV window mode (no Streamlit).
         Use dashboard.py for the full dashboard experience.

Counting: vehicle_count = number of YOLO detection boxes in this frame.
          Same number as what you see drawn on screen.
"""

import os, sys
from pathlib import Path
from ultralytics import YOLO
import cv2
from tracker import tracker
from signal_logic import get_signal_time, density_predictor

VIDEO_FOLDER = Path("videos")
KNOWN_VIDEOS = [
    "14552311-hd_1920_1080_50fps.mp4",
    "15300538-hd_1920_1080_60fps.mp4",
]

def find_video() -> str:
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        return sys.argv[1]
    for name in KNOWN_VIDEOS:
        p = VIDEO_FOLDER / name
        if p.exists():
            print(f"[INFO] Using video: {p}")
            return str(p)
    if VIDEO_FOLDER.exists():
        for f in VIDEO_FOLDER.iterdir():
            if f.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
                return str(f)
    print("[ERROR] No video found in videos/ folder.")
    sys.exit(1)

video_path = find_video()
model = YOLO("yolov8s.pt")  # s=small: much better motorcycle detection than nano
cap   = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print(f"[ERROR] Cannot open: {video_path}")
    sys.exit(1)

VEHICLE_CLASSES = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck
VEHICLE_WEIGHTS = {"bicycle": 0.3, "car": 1.0, "motorcycle": 0.5, "bus": 2.0, "truck": 2.0}

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Video ended.")
        break

    results = model(frame, conf=0.20, iou=0.40, imgsz=1280, verbose=False)[0]
    detections    = []
    vehicle_types = {}
    total_weight  = 0.0
    vehicle_count = 0   # counts YOLO boxes only

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls not in VEHICLE_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf  = float(box.conf[0])
        name  = model.names[cls]
        vehicle_count += 1
        vehicle_types[name] = vehicle_types.get(name, 0) + 1
        total_weight += VEHICLE_WEIGHTS.get(name, 1.0)
        # Per-class confidence filter
        per_class_min = {1: 0.18, 2: 0.40, 3: 0.18, 5: 0.40, 7: 0.40}
        if conf < per_class_min.get(cls, 0.40):
            continue
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 80), 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(y1-6,14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 80), 1, cv2.LINE_AA)

    tracks = tracker.update_tracks(detections, frame=frame)
    for track in tracks:
        if not track.is_confirmed():
            continue
        l, t, r, b = map(int, track.to_ltrb())
        cv2.rectangle(frame, (l, t), (r, b), (255, 140, 0), 2)
        cv2.putText(frame, f"#{track.track_id}", (l, max(t-6,14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 140, 0), 1, cv2.LINE_AA)

    green_time, level = get_signal_time(vehicle_count)
    density_predictor.update(total_weight)
    pred_density, pred_level, pred_conf = density_predictor.predict()

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (310, 135), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for text, pos, color in [
        (f"Vehicles : {vehicle_count}", (12, 28),  (0, 220, 255)),
        (f"Traffic  : {level}",         (12, 56),  (255, 220, 0)),
        (f"Green    : {green_time}s",   (12, 84),  (0, 255, 120)),
        (f"Pred     : {pred_level}",    (12, 112), (200, 100, 255)),
    ]:
        cv2.putText(frame, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)

    cv2.imshow("AI Traffic System", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

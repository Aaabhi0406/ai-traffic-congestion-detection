# Architecture

Every number displayed on the dashboard or CLI overlay comes from one function: `TrafficPipeline.process_frame()` in `src/traffic_system/pipeline.py`. This document walks through what happens to a frame, stage by stage.

## 1. ROI (Region of Interest)

A trapezoid polygon is built for the frame's dimensions (`pipeline.build_roi`), covering the visible road from near-top-of-frame to near-bottom, with the top edge inset slightly to account for perspective (the far lane appears narrower than the near one). Only vehicles whose bounding-box centre falls inside this polygon are counted; vehicles detected outside it are still drawn (in grey) but ignored for counting and signal timing.

If your camera's angle or framing differs, adjust the `config.ROI_*` fractions rather than hardcoding new pixel coordinates — the fractions keep the ROI valid across different input resolutions.

## 2. YOLOv8 detection

`detection.detect_vehicles()` runs the model at `imgsz=1280` (higher than YOLO's default 640, so small/far vehicles are still resolvable) and applies:
- a class filter (bicycle, car, motorcycle, bus, truck — the five COCO vehicle classes we care about)
- a **per-class confidence threshold** (`config.CONF_PER_CLASS`) — see the README's "Key engineering decisions" for why this is per-class rather than global
- a box-size filter, rejecting anything smaller than 20×20px (noise) or larger than 85% of the frame in either dimension (DeepSort ghost tracks occasionally produce near-full-frame boxes)

## 3. ROI filter

For each surviving detection, its box centre is tested against the ROI polygon with `cv2.pointPolygonTest`. Inside → counted, drawn green. Outside → drawn grey, not counted.

## 4. DeepSort tracking

Only ROI-filtered detections are passed to the tracker, so tracking IDs are never assigned to vehicles outside the region we care about. The tracker's parameters (`config.DEEPSORT_*`) were tuned down from DeepSort's defaults after ghost-track issues on the test footage — see the README for specifics. A track is only drawn if a real detection is still nearby (`tracking.has_nearby_detection`), which further suppresses tracks that persist briefly with no supporting evidence.

Tracking IDs are used for line-crossing counting and the on-screen `#ID` label — **not** for the per-frame `vehicle_count` used in signal timing, which comes directly from ROI-filtered detections each frame.

## 5. Line-crossing counter

A horizontal line sits at 75% of the frame height, inside the ROI. `tracking.LineCrossingCounter` remembers each track's vertical centre from frame to frame and records a crossing when a track moves from above the line to below it (top → bottom only — the assumption is traffic flowing in one direction through frame). This produces a cumulative throughput count, separate from the instantaneous `vehicle_count`.

## 6. Signal logic

`signal_logic.get_signal_time(vehicle_count)` maps the ROI vehicle count for this frame to a traffic level and a green-light duration:

| Level | Vehicle count | Green time |
|---|---|---|
| LOW | 0 – `config.LOW_MAX` | 15s |
| MEDIUM | `config.LOW_MAX + 1` – `config.MEDIUM_MAX` | 25s |
| HIGH | above `config.MEDIUM_MAX` | 40s |

These are simple threshold bands rather than a continuous function, which keeps the signal behavior predictable and easy to explain/justify to a non-ML stakeholder — an important property for anything that would ever touch real infrastructure.

## 7. Random Forest density prediction

`signal_logic.TrafficDensityPredictor` maintains a rolling history of *weighted* density (each detected vehicle contributes `config.VEHICLE_WEIGHTS[class]` rather than a flat 1, so a truck counts for more than a bicycle). Every frame it:

1. Appends the current weighted density to its history.
2. Once enough history exists, builds a training pair: a feature vector from a 30-frame window, paired with the actual density 15 frames later.
3. Retrains a `RandomForestRegressor` every 30 new samples, once at least 80 have accumulated.
4. On request, predicts density 15 frames ahead from the most recent 30-frame window, and reports its R² from the last fit as a confidence value.

Before the model has enough data to train, `predict()` falls back to an exponential moving average of the current window, so the dashboard always has *something* to show — just flagged as "Warming up…" rather than presented with false confidence.

## 8. Output

All eight numbers above are packaged into a single dict returned by `process_frame()`. Both `cli.py` and `dashboard.py` read from this same dict — there's exactly one place vehicle counting/timing logic lives, which is the main fix from the previous version of this repo (where `main.py` had a second, slightly different copy of the detection logic).

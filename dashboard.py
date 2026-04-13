"""
dashboard.py — Streamlit Traffic Management Dashboard

Run with:
    streamlit run dashboard.py

Features:
  • BOTH videos available directly in the sidebar — no manual typing needed
  • Auto-scans the videos/ folder for any additional video files
  • Live video feed with vehicle detection + tracking overlays
  • Real-time vehicle count, type breakdown, weighted density
  • Adaptive green light time recommendation
  • ML-predicted future traffic density (Random Forest)
  • Rolling charts for density history
  • Traffic level status card
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
from collections import deque
from pathlib import Path

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Traffic Management Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background: #0a0d14;
    color: #e8eaf0;
}

.stApp { background: #0a0d14; }

.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2035 100%);
    border: 1px solid #2a3550;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, #00e5ff);
}
.metric-card .label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #6b7a9a;
    margin-bottom: 6px;
}
.metric-card .value {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1;
    color: var(--accent, #00e5ff);
}
.metric-card .sub {
    font-size: 0.75rem;
    color: #8892aa;
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
}

.level-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.1em;
}
.level-LOW    { background: #0d3320; color: #00ff88; border: 1px solid #00ff8844; }
.level-MEDIUM { background: #3a2900; color: #ffaa00; border: 1px solid #ffaa0044; }
.level-HIGH   { background: #3a0d0d; color: #ff4444; border: 1px solid #ff444444; }
.level-UNKNOWN{ background: #1a1f2e; color: #8892aa; border: 1px solid #2a3550; }

.section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #4a5568;
    margin: 18px 0 10px;
    border-left: 3px solid #2a3550;
    padding-left: 8px;
}

.signal-container {
    background: #111827;
    border: 1px solid #2a3550;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.signal-light {
    width: 56px;
    height: 56px;
    border-radius: 50%;
    margin: 8px auto;
    transition: all 0.3s;
}
.light-red    { background: #ff2222; box-shadow: 0 0 20px #ff222288; }
.light-yellow { background: #ffcc00; box-shadow: 0 0 20px #ffcc0088; }
.light-green  { background: #00ff66; box-shadow: 0 0 20px #00ff6688; }
.light-off    { background: #222; box-shadow: none; }

.conf-bar-wrap {
    background: #1a2035;
    border-radius: 4px;
    height: 8px;
    margin-top: 8px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #7c3aed, #a855f7);
    transition: width 0.4s ease;
}

/* Video selector cards */
.video-card {
    background: #111827;
    border: 1px solid #2a3550;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 8px;
    cursor: pointer;
    transition: border-color 0.2s;
}
.video-card:hover { border-color: #00e5ff55; }
.video-card.selected { border-color: #00e5ff; background: #0d1f33; }
.video-card .vc-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #00e5ff;
    word-break: break-all;
}
.video-card .vc-info {
    font-size: 0.68rem;
    color: #4a5568;
    margin-top: 3px;
}

.stPlotlyChart, .stDataFrame { border-radius: 12px; overflow: hidden; }
section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #1e2640; }
</style>
""", unsafe_allow_html=True)


# ─── Helper: scan available videos ───────────────────────────────────────────
KNOWN_VIDEOS = {
    "14552311-hd_1920_1080_50fps.mp4": "Video 1 · 50fps · 1080p",
    "15300538-hd_1920_1080_60fps.mp4": "Video 2 · 60fps · 1080p",
}

def scan_videos() -> list[dict]:
    """
    Return list of dicts: {path, label, exists}
    - First lists the two known videos (whether they exist or not)
    - Then appends any extra .mp4/.avi/.mov files found in videos/
    """
    video_dir = Path("videos")
    result = []
    seen = set()

    for fname, label in KNOWN_VIDEOS.items():
        p = video_dir / fname
        result.append({"path": str(p), "label": label, "exists": p.exists(), "fname": fname})
        seen.add(fname)

    # Pick up any extra videos the user dropped in
    if video_dir.exists():
        for f in sorted(video_dir.iterdir()):
            if f.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv") and f.name not in seen:
                result.append({
                    "path": str(f),
                    "label": f"Extra · {f.name}",
                    "exists": True,
                    "fname": f.name,
                })

    return result


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚦 AI Traffic System")
    st.markdown("<div class='section-title'>Select Video</div>", unsafe_allow_html=True)

    videos = scan_videos()
    available = [v for v in videos if v["exists"]]

    if not available:
        st.error("No video files found in the `videos/` folder.\nAdd at least one .mp4 file and refresh.")
        st.stop()

    # Build radio options (show all available)
    video_labels = [f"📹 {v['label']}\n`{v['fname']}`" for v in available]
    chosen_idx = st.radio(
        "Available videos",
        range(len(available)),
        format_func=lambda i: available[i]["label"],
        index=0,
    )
    video_source = available[chosen_idx]["path"]

    # Show info about chosen file
    chosen = available[chosen_idx]
    p = Path(chosen["path"])
    size_mb = p.stat().st_size / (1024 * 1024) if p.exists() else 0
    st.markdown(f"""
<div style='background:#0d1117;border:1px solid #1e2640;border-radius:8px;padding:10px 12px;margin-top:6px'>
  <div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#4a5568'>SELECTED FILE</div>
  <div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#00e5ff;word-break:break-all;margin-top:4px'>{chosen['fname']}</div>
  <div style='font-size:0.68rem;color:#6b7a9a;margin-top:3px'>{size_mb:.1f} MB</div>
</div>
""", unsafe_allow_html=True)

    # Show missing videos notice
    missing = [v for v in videos if not v["exists"]]
    if missing:
        with st.expander("⚠ Missing videos"):
            for m in missing:
                st.markdown(f"<span style='color:#ff4444;font-size:0.7rem'>`{m['fname']}`</span>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Model & Settings</div>", unsafe_allow_html=True)
    model_path = st.selectbox("YOLO Model", ["yolov8s.pt", "yolov8m.pt", "yolov8n.pt"], index=0)
    conf_threshold = st.slider("Detection Confidence", 0.1, 0.9, 0.3, 0.05)
    process_every = st.slider("Process every N frames", 1, 5, 1)
    max_history = st.slider("Chart history (frames)", 50, 300, 100, 10)

    st.markdown("<div class='section-title'>System Architecture</div>", unsafe_allow_html=True)
    st.markdown("""
<small style='color:#6b7a9a'>
<b style='color:#00e5ff'>① ROI (Region of Interest)</b><br>
Trapezoid zone on video. Only vehicles whose centre falls inside are counted (green boxes).<br><br>
<b style='color:#00d4ff'>② Counting Line</b><br>
Cyan horizontal line inside ROI. Counts every vehicle that crosses top→bottom (throughput).<br><br>
<b style='color:#ffaa00'>③ Per-class Detection</b><br>
Car/bus/truck: conf ≥ 0.40 · Motorcycle/Bicycle: conf ≥ 0.18. Runs at imgsz=1280 so small vehicles appear larger in model input → far more detections.<br><br>
<b style='color:#ff9900'>④ DeepSort Tracker</b><br>
Assigns persistent orange #ID labels. Does NOT affect vehicle count.<br><br>
<b style='color:#a855f7'>⑤ Random Forest Prediction</b><br>
Predicts density 15 frames (~0.5s) ahead. Features: 30-frame rolling window of weighted density (mean, std, slope, rate-of-change).
</small>
""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Controls</div>", unsafe_allow_html=True)
    start_btn = st.button("▶ Start Processing", use_container_width=True, type="primary")
    stop_btn  = st.button("■ Stop", use_container_width=True)


# ─── Main layout ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex;align-items:center;gap:12px;margin-bottom:4px'>
  <span style='font-size:1.8rem'>🚦</span>
  <div>
    <div style='font-size:1.5rem;font-weight:800;letter-spacing:-0.02em'>AI Traffic Management System</div>
    <div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#4a5568;letter-spacing:0.1em'>
      YOLOv8 · DeepSort · Random Forest Density Predictor
    </div>
  </div>
</div>
<hr style='border-color:#1e2640;margin:12px 0 20px'>
""", unsafe_allow_html=True)

col_video, col_stats = st.columns([3, 2])

with col_video:
    video_placeholder = st.empty()
    status_placeholder = st.empty()

with col_stats:
    st.markdown("<div class='section-title'>Live Metrics — ROI This Frame</div>", unsafe_allow_html=True)

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        vehicles_ph = st.empty()   # vehicles inside ROI this frame
    with m_col2:
        weight_ph = st.empty()     # traffic level

    m_col3, m_col4 = st.columns(2)
    with m_col3:
        crossing_ph = st.empty()   # line-crossing total
    with m_col4:
        signal_ph = st.empty()

    st.markdown("<div class='section-title'>Prediction (Random Forest)</div>", unsafe_allow_html=True)
    pred_ph = st.empty()

    st.markdown("<div class='section-title'>Vehicle Breakdown — ROI This Frame</div>", unsafe_allow_html=True)
    breakdown_ph = st.empty()

st.markdown("<div class='section-title'>Traffic Density History</div>", unsafe_allow_html=True)
chart_ph = st.empty()

# ─── Session state ───────────────────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False
if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=300)
if "last_video" not in st.session_state:
    st.session_state.last_video = None

if start_btn:
    # Reset history when switching videos
    if st.session_state.last_video != video_source:
        st.session_state.history = deque(maxlen=300)
        st.session_state.last_video = video_source
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False


# ─── UI helpers ──────────────────────────────────────────────────────────────
def level_badge(level: str) -> str:
    return f"<span class='level-badge level-{level}'>{level}</span>"


def metric_card(label: str, value: str, sub: str = "", accent: str = "#00e5ff") -> str:
    return f"""
<div class='metric-card' style='--accent:{accent}'>
  <div class='label'>{label}</div>
  <div class='value'>{value}</div>
  {"<div class='sub'>" + sub + "</div>" if sub else ""}
</div>"""


def signal_panel(level: str, green_time: int) -> str:
    if level == "LOW":
        r, y, g = "light-off", "light-off", "light-green"
    elif level == "MEDIUM":
        r, y, g = "light-off", "light-yellow", "light-off"
    elif level == "HIGH":
        r, y, g = "light-red", "light-off", "light-off"
    else:
        r, y, g = "light-off", "light-off", "light-off"

    return f"""
<div class='signal-container'>
  <div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#4a5568;letter-spacing:0.15em;text-transform:uppercase'>Signal State</div>
  <div class='signal-light {r}'></div>
  <div class='signal-light {y}'></div>
  <div class='signal-light {g}'></div>
  <div style='font-size:1.5rem;font-weight:800;color:#00e5ff;margin-top:8px'>{green_time}s</div>
  <div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#6b7a9a'>green time</div>
</div>"""


def pred_panel(pred_density: float, pred_level: str, confidence: float, trained: bool) -> str:
    conf_pct = int(confidence * 100)
    status = "RF Model Active" if trained else "Warming up…"
    return f"""
<div class='metric-card' style='--accent:#a855f7'>
  <div style='display:flex;justify-content:space-between;align-items:center'>
    <div class='label'>Predicted Density (0.5s ahead)</div>
    <div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#a855f7'>{status}</div>
  </div>
  <div style='display:flex;align-items:baseline;gap:12px;margin-top:4px'>
    <div class='value' style='color:#a855f7'>{pred_density:.1f}</div>
    {level_badge(pred_level)}
  </div>
  <div class='conf-bar-wrap'>
    <div class='conf-bar-fill' style='width:{conf_pct}%'></div>
  </div>
  <div class='sub'>R² confidence: {conf_pct}%</div>
</div>"""


# ─── Processing loop ─────────────────────────────────────────────────────────
if st.session_state.running:
    from traffic_core import get_model, process_frame
    from signal_logic import density_predictor

    src = 0 if "Webcam" in str(video_source) else video_source
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        status_placeholder.error(f"❌ Cannot open video: {src}\n\nMake sure the file exists in the `videos/` folder.")
        st.session_state.running = False
    else:
        model = get_model(model_path)
        frame_idx = 0
        video_name = Path(str(src)).name if src != 0 else "webcam"
        status_placeholder.info(f"▶ Processing: **{video_name}**")

        try:
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    status_placeholder.info("📼 Video ended — restarting…")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                frame_idx += 1
                if frame_idx % process_every != 0:
                    continue

                result = process_frame(frame.copy(), model)

                st.session_state.history.append({
                    "frame": frame_idx,
                    "density": result["total_weight"],
                    "pred_density": result["pred_density"],
                    "count": result["vehicle_count"],
                })

                annotated = cv2.cvtColor(result["annotated_frame"], cv2.COLOR_BGR2RGB)
                video_placeholder.image(annotated, channels="RGB", use_container_width=True)

                vehicles_ph.markdown(
                    metric_card("Vehicles in ROI", str(result["vehicle_count"]),
                                "green boxes on screen", "#00e5ff"),
                    unsafe_allow_html=True)

                weight_ph.markdown(
                    metric_card("Traffic Level", result["traffic_level"],
                                f"green: {result['green_time']}s",
                                "#00ff88" if result["traffic_level"] == "LOW"
                                else "#ffaa00" if result["traffic_level"] == "MEDIUM"
                                else "#ff4444"),
                    unsafe_allow_html=True)

                crossing_ph.markdown(
                    metric_card("Line Crossings", str(result["crossing_count"]),
                                "vehicles crossed line", "#00d4ff"),
                    unsafe_allow_html=True)

                signal_ph.markdown(
                    signal_panel(result["traffic_level"], result["green_time"]),
                    unsafe_allow_html=True)

                pred_ph.markdown(
                    pred_panel(result["pred_density"], result["pred_level"],
                               result["pred_confidence"],
                               density_predictor._is_trained),
                    unsafe_allow_html=True)

                if result["vehicle_types"]:
                    df_types = pd.DataFrame(
                        list(result["vehicle_types"].items()),
                        columns=["Type", "Count"]
                    ).sort_values("Count", ascending=False)
                    breakdown_ph.dataframe(
                        df_types, hide_index=True, use_container_width=True
                    )

                if len(st.session_state.history) > 5:
                    df_hist = pd.DataFrame(list(st.session_state.history))
                    df_hist = df_hist.tail(max_history)
                    chart_ph.line_chart(
                        df_hist.set_index("frame")[["density", "pred_density"]],
                        color=["#00e5ff", "#a855f7"]
                    )

                time.sleep(0.01)

        finally:
            cap.release()

else:
    # ── Idle state ──
    video_placeholder.markdown("""
<div style='height:360px;display:flex;flex-direction:column;align-items:center;
            justify-content:center;background:#111827;border-radius:12px;
            border:1px dashed #2a3550'>
  <div style='font-size:3rem;margin-bottom:12px'>🚦</div>
  <div style='font-family:JetBrains Mono,monospace;color:#4a5568;font-size:0.8rem;
              letter-spacing:0.1em'>SELECT A VIDEO AND PRESS START</div>
</div>""", unsafe_allow_html=True)

    vehicles_ph.markdown(metric_card("Vehicles in ROI", "—", "green boxes on screen"), unsafe_allow_html=True)
    weight_ph.markdown(metric_card("Traffic Level", "—", "waiting"), unsafe_allow_html=True)
    crossing_ph.markdown(metric_card("Line Crossings", "—", "vehicles crossed line", "#00d4ff"), unsafe_allow_html=True)
    signal_ph.markdown(signal_panel("UNKNOWN", 0), unsafe_allow_html=True)
    pred_ph.markdown(pred_panel(0, "UNKNOWN", 0, False), unsafe_allow_html=True)
    breakdown_ph.markdown("<div style='color:#4a5568;font-size:0.8rem'>No data yet</div>",
                          unsafe_allow_html=True)

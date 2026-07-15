"""
cli.py — OpenCV window mode (no Streamlit).

Use `streamlit run src/traffic_system/dashboard.py` for the full dashboard.

This used to be main.py, with its own copy of the detection/counting logic
at slightly different confidence thresholds than traffic_core.py. It now
just wires video I/O around the shared TrafficPipeline, so CLI and
dashboard modes are guaranteed to produce identical counts.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

from . import config
from .detection import get_model
from .pipeline import TrafficPipeline

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VIDEO_FOLDER = Path("videos")


def find_video(explicit_path: str | None) -> str:
    if explicit_path and Path(explicit_path).exists():
        return explicit_path

    for name in config.KNOWN_VIDEOS:
        p = VIDEO_FOLDER / name
        if p.exists():
            logger.info("Using video: %s", p)
            return str(p)

    if VIDEO_FOLDER.exists():
        for f in VIDEO_FOLDER.iterdir():
            if f.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
                logger.info("Using video: %s", f)
                return str(f)

    logger.error("No video found in %s/ and no valid path given.", VIDEO_FOLDER)
    sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AI Traffic Management System — OpenCV window mode")
    parser.add_argument("video", nargs="?", default=None, help="Path to a video file (optional)")
    parser.add_argument("--model", default=config.DEFAULT_MODEL_PATH, help="YOLO model path/name")
    args = parser.parse_args(argv)

    video_path = find_video(args.video)
    model = get_model(args.model)
    pipeline = TrafficPipeline()

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        sys.exit(1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("Video ended.")
                break

            result = pipeline.process_frame(frame, model)
            cv2.imshow("AI Traffic System", result["annotated_frame"])
            if cv2.waitKey(1) & 0xFF == 27:  # Esc to quit
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

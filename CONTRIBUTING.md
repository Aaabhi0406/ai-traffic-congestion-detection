# Contributing

This started as a personal portfolio/learning project, but issues and PRs are
welcome.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                # installs the traffic_system package in editable mode
```

## Running tests

```bash
pip install pytest ruff
ruff check src/ tests/
pytest tests/ -v
```

Tests that need a live video feed, YOLO weights, or DeepSort are out of scope
for the automated suite — CI only runs the pure-logic tests (signal timing,
ROI geometry, feature engineering, line-crossing counting). If you're adding
detection/tracking behavior, please add a unit test that exercises the logic
through a stub model or fake track object rather than a real video, so CI
stays fast and reproducible (see `tests/test_detection.py` for the pattern).

## Code style

- Type hints and docstrings on public functions/classes.
- Tunable constants belong in `src/traffic_system/config.py`, not scattered
  magic numbers.
- Logging (`logging` module) instead of `print()` for anything beyond a
  simple CLI message.

## Reporting bugs

Open an issue with: what you ran, what you expected, what happened instead,
and — if it's detection/tracking related — the video/frame where it
reproduces if you can share it.

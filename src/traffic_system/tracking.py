"""
tracking.py — DeepSort wrapper + line-crossing counter.

Both pieces of state (the tracker itself and the line-crossing history) are
encapsulated in classes instead of module-level globals. The previous
version kept `_prev_centres` / `crossing_count` as module globals, which
works for a single-process script but isn't resettable between runs/tests
and would be a footgun the moment this ran anywhere concurrent.
"""

from __future__ import annotations

from . import config


def build_tracker():
    """Construct a DeepSort tracker with our tuned parameters."""
    from deep_sort_realtime.deepsort_tracker import DeepSort  # lazy import

    return DeepSort(
        max_age=config.DEEPSORT_MAX_AGE,
        n_init=config.DEEPSORT_N_INIT,
        nms_max_overlap=config.DEEPSORT_NMS_MAX_OVERLAP,
        max_cosine_distance=config.DEEPSORT_MAX_COSINE_DISTANCE,
        nn_budget=config.DEEPSORT_NN_BUDGET,
    )


def has_nearby_detection(
    lx: float, ty: float, rx: float, by: float,
    centres: list[tuple[float, float]],
    threshold: float = 60,
) -> bool:
    """True if any detection centre falls inside (or just outside) this track box.

    Used to suppress DeepSort "ghost" tracks that persist for a few frames
    with no supporting detection.
    """
    for cx, cy in centres:
        if lx - threshold < cx < rx + threshold and ty - threshold < cy < by + threshold:
            return True
    return False


class LineCrossingCounter:
    """Tracks each confirmed track's vertical centre frame-to-frame and
    counts top-to-bottom crossings of a horizontal counting line."""

    def __init__(self) -> None:
        self._prev_centres: dict[int, float] = {}
        self.total: int = 0

    def reset(self) -> None:
        self._prev_centres.clear()
        self.total = 0

    def update(self, track_id: int, centre_y: float, line_y: float) -> bool:
        """Record this track's centre for this frame; return True if it just
        crossed the line top -> bottom."""
        crossed = False
        prev_cy = self._prev_centres.get(track_id)
        if prev_cy is not None and prev_cy < line_y <= centre_y:
            self.total += 1
            crossed = True
        return crossed

    def commit_frame(self, curr_centres: dict[int, float]) -> None:
        """Call once per frame after all update() calls, with the full set of
        this frame's confirmed track centres, to roll state forward."""
        self._prev_centres = curr_centres

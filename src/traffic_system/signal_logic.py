"""
signal_logic.py — adaptive signal timing + Random Forest density predictor.

TRAFFIC LEVEL — based on vehicles counted INSIDE the ROI this frame:
─────────────────────────────────────────────────────────────────────
  LOW    : 0  – config.LOW_MAX        vehicles -> 15s green
  MEDIUM : config.LOW_MAX+1 – config.MEDIUM_MAX -> 25s green
  HIGH   : > config.MEDIUM_MAX        vehicles -> 40s green

NOTE: an earlier version of this docstring (and traffic_core.py's) quoted
bands of 0-6 / 7-12 / 13+, which no longer matched the tuned constants
(LOW_MAX=12, MEDIUM_MAX=25). The bands above reflect the actual thresholds
in config.py — if the intent was the tighter 0-6/7-12/13+ split, update
config.LOW_MAX / config.MEDIUM_MAX rather than this docstring.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

from . import config

logger = logging.getLogger(__name__)


def get_signal_time(vehicle_count: int) -> tuple[int, str]:
    """Return (green_seconds, level) for a given ROI vehicle count."""
    if vehicle_count <= config.LOW_MAX:
        level = "LOW"
    elif vehicle_count <= config.MEDIUM_MAX:
        level = "MEDIUM"
    else:
        level = "HIGH"
    return config.GREEN_TIME[level], level


class TrafficDensityPredictor:
    """
    Random Forest Regressor — predicts weighted vehicle density N frames ahead.

    DATA FLOW:
      update(total_weight) is called every frame:
        - appends to a rolling history of weighted density values
        - once enough history exists, builds a (X, y) training pair:
              X = feature vector from a window of `window_size` past values
              y = the actual density `predict_horizon` frames later
        - retrains every `retrain_every` new samples, once
          `min_samples` have been collected

      predict() is called whenever a prediction is needed:
        - uses the most recent `window_size` values as features
        - if trained, the RF model predicts density `predict_horizon` frames
          ahead; otherwise falls back to an EMA of the window
        - returns (predicted_density, predicted_level, R² confidence)
    """

    def __init__(
        self,
        window_size: int = config.RF_WINDOW_SIZE,
        predict_horizon: int = config.RF_PREDICT_HORIZON,
        min_samples: int = config.RF_MIN_TRAIN_SAMPLES,
        retrain_every: int = config.RF_RETRAIN_EVERY,
        max_train_history: int = config.RF_MAX_TRAIN_HISTORY,
    ):
        self.window_size = window_size
        self.predict_horizon = predict_horizon
        self.min_samples = min_samples
        self.retrain_every = retrain_every
        self.max_train_history = max_train_history

        self.history: deque[float] = deque(
            maxlen=window_size * 2 + predict_horizon + 10
        )
        self.X_train: list[list[float]] = []
        self.y_train: list[float] = []
        self.model = None
        self._is_trained = False
        self.confidence = 0.0

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def _features(self, window: list[float]) -> list[float]:
        """Feature vector from a window of density values: raw values + mean
        + std + min + max + linear slope + short-term rate-of-change."""
        arr = np.array(window, dtype=float)
        feat = list(arr)
        feat += [
            float(np.mean(arr)),
            float(np.std(arr)),
            float(np.min(arr)),
            float(np.max(arr)),
        ]
        slope = (
            float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
            if len(arr) > 1
            else 0.0
        )
        feat.append(slope)
        feat.append(float(arr[-1] - arr[-5]) if len(arr) >= 5 else 0.0)
        return feat

    def _to_level(self, density: float) -> str:
        if density < config.DENSITY_LOW:
            return "LOW"
        elif density < config.DENSITY_MEDIUM:
            return "MEDIUM"
        return "HIGH"

    def update(self, weighted_count: float) -> None:
        self.history.append(weighted_count)
        hist = list(self.history)
        n = len(hist)
        if n >= self.window_size + self.predict_horizon + 1:
            idx = n - self.predict_horizon - 1
            window = hist[idx - self.window_size : idx]
            target = hist[idx + self.predict_horizon - 1]
            if len(window) == self.window_size:
                self.X_train.append(self._features(window))
                self.y_train.append(target)

        if (
            len(self.X_train) >= self.min_samples
            and len(self.X_train) % self.retrain_every == 0
        ):
            self._train()

    def _train(self) -> None:
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            logger.warning("scikit-learn not available; skipping RF training")
            return

        X = np.array(self.X_train[-self.max_train_history :])
        y = np.array(self.y_train[-self.max_train_history :])
        rf = RandomForestRegressor(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            random_state=config.RF_RANDOM_STATE,
            n_jobs=-1,
        )
        rf.fit(X, y)
        self.model = rf
        self._is_trained = True

        pred = rf.predict(X)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.confidence = float(max(0.0, min(1.0, 1 - ss_res / (ss_tot + 1e-9))))
        logger.info("RF retrained on %d samples, R^2=%.3f", len(X), self.confidence)

    def predict(self) -> tuple[float, str, float]:
        hist = list(self.history)
        if len(hist) < self.window_size:
            avg = float(np.mean(hist)) if hist else 0.0
            return avg, self._to_level(avg), 0.0

        window = hist[-self.window_size :]
        features = self._features(window)

        if self._is_trained and self.model is not None:
            pred = float(max(0.0, self.model.predict([features])[0]))
        else:
            alpha, ema = 0.3, window[0]
            for v in window[1:]:
                ema = alpha * v + (1 - alpha) * ema
            pred = ema

        return pred, self._to_level(pred), self.confidence

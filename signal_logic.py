"""
signal_logic.py — Signal Logic + Random Forest Density Predictor

TRAFFIC LEVEL — based on vehicles INSIDE ROI this frame:
─────────────────────────────────────────────────────────
  LOW    : 0 – 6  vehicles  → 15s green
  MEDIUM : 7 – 12 vehicles  → 25s green
  HIGH   : 13+    vehicles  → 40s green

These match exactly what is drawn on the video (green boxes inside the ROI).
"""

import numpy as np
from collections import deque

LOW_MAX    =  12
MEDIUM_MAX = 25

GREEN_TIME = {"LOW": 15, "MEDIUM": 25, "HIGH": 40}

# Density thresholds for ML predictor level labels
DENSITY_LOW    =  6.0
DENSITY_MEDIUM = 14.0


def get_signal_time(vehicle_count: int) -> tuple[int, str]:
    """
    Return (green_seconds, level).
    vehicle_count = number of green boxes visible inside ROI on screen.
    """
    if vehicle_count <= LOW_MAX:
        level = "LOW"
    elif vehicle_count <= MEDIUM_MAX:
        level = "MEDIUM"
    else:
        level = "HIGH"
    return GREEN_TIME[level], level


class TrafficDensityPredictor:
    """
    Random Forest Regressor — predicts weighted vehicle density 15 frames ahead.

    DATA FLOW:
      process_frame() calls update(total_weight) every frame
          → builds rolling history of weighted density values
          → builds (X, y) training pairs:
              X = feature vector from window of 30 past density values
              y = actual density 15 frames later
          → retrains every 30 new samples (once 80 samples collected)

      process_frame() calls predict()
          → uses last 30 density values as features
          → RF model outputs predicted density
          → density mapped to LOW/MEDIUM/HIGH level
          → returns (pred_density, pred_level, R² confidence)
    """

    def __init__(self, window_size: int = 30, predict_horizon: int = 15):
        self.window_size     = window_size
        self.predict_horizon = predict_horizon
        self.history: deque[float] = deque(
            maxlen=window_size * 2 + predict_horizon + 10)
        self.X_train: list[list[float]] = []
        self.y_train: list[float]       = []
        self.model       = None
        self._is_trained = False
        self.min_samples = 80
        self.confidence  = 0.0

    def _features(self, window: list[float]) -> list[float]:
        """
        Feature vector from a window of density values:
          raw values + mean + std + min + max + linear slope + rate-of-change
        """
        arr  = np.array(window, dtype=float)
        feat = list(arr)
        feat += [float(np.mean(arr)), float(np.std(arr)),
                 float(np.min(arr)),  float(np.max(arr))]
        slope = (float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
                 if len(arr) > 1 else 0.0)
        feat.append(slope)
        feat.append(float(arr[-1] - arr[-5]) if len(arr) >= 5 else 0.0)
        return feat

    def _to_level(self, density: float) -> str:
        if density < DENSITY_LOW:
            return "LOW"
        elif density < DENSITY_MEDIUM:
            return "MEDIUM"
        return "HIGH"

    def update(self, weighted_count: float) -> None:
        self.history.append(weighted_count)
        hist = list(self.history)
        n = len(hist)
        if n >= self.window_size + self.predict_horizon + 1:
            idx    = n - self.predict_horizon - 1
            window = hist[idx - self.window_size: idx]
            target = hist[idx + self.predict_horizon - 1]
            if len(window) == self.window_size:
                self.X_train.append(self._features(window))
                self.y_train.append(target)
        if (len(self.X_train) >= self.min_samples
                and len(self.X_train) % 30 == 0):
            self._train()

    def _train(self) -> None:
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            return
        X = np.array(self.X_train[-500:])
        y = np.array(self.y_train[-500:])
        rf = RandomForestRegressor(
            n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        self.model       = rf
        self._is_trained = True
        pred   = rf.predict(X)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.confidence = float(
            max(0.0, min(1.0, 1 - ss_res / (ss_tot + 1e-9))))

    def predict(self) -> tuple[float, str, float]:
        hist = list(self.history)
        if len(hist) < self.window_size:
            avg = float(np.mean(hist)) if hist else 0.0
            return avg, self._to_level(avg), 0.0
        window   = hist[-self.window_size:]
        features = self._features(window)
        if self._is_trained and self.model is not None:
            pred = float(max(0.0, self.model.predict([features])[0]))
        else:
            alpha, ema = 0.3, window[0]
            for v in window[1:]:
                ema = alpha * v + (1 - alpha) * ema
            pred = ema
        return pred, self._to_level(pred), self.confidence


density_predictor = TrafficDensityPredictor(window_size=30, predict_horizon=15)

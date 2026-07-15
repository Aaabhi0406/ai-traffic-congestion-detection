"""Tests for get_signal_time() and the TrafficDensityPredictor's pure-logic
pieces. None of this needs a live video feed, YOLO, or DeepSort."""

import numpy as np
import pytest

from traffic_system import config
from traffic_system.signal_logic import TrafficDensityPredictor, get_signal_time


class TestGetSignalTime:
    def test_zero_vehicles_is_low(self):
        green_time, level = get_signal_time(0)
        assert level == "LOW"
        assert green_time == config.GREEN_TIME["LOW"]

    def test_low_boundary_is_still_low(self):
        _, level = get_signal_time(config.LOW_MAX)
        assert level == "LOW"

    def test_just_above_low_boundary_is_medium(self):
        _, level = get_signal_time(config.LOW_MAX + 1)
        assert level == "MEDIUM"

    def test_medium_boundary_is_still_medium(self):
        _, level = get_signal_time(config.MEDIUM_MAX)
        assert level == "MEDIUM"

    def test_just_above_medium_boundary_is_high(self):
        _, level = get_signal_time(config.MEDIUM_MAX + 1)
        assert level == "HIGH"

    def test_large_count_is_high(self):
        _, level = get_signal_time(500)
        assert level == "HIGH"

    @pytest.mark.parametrize("level", ["LOW", "MEDIUM", "HIGH"])
    def test_green_time_matches_config(self, level):
        counts = {"LOW": 0, "MEDIUM": config.LOW_MAX + 1, "HIGH": config.MEDIUM_MAX + 1}
        green_time, returned_level = get_signal_time(counts[level])
        assert returned_level == level
        assert green_time == config.GREEN_TIME[level]

    def test_returns_int_seconds_and_str_level(self):
        green_time, level = get_signal_time(5)
        assert isinstance(green_time, int)
        assert isinstance(level, str)


class TestDensityPredictorFeatures:
    """_features() is the exact vector fed to the Random Forest — a bug here
    silently corrupts every prediction, so it gets the most scrutiny."""

    def setup_method(self):
        self.predictor = TrafficDensityPredictor()

    def test_feature_vector_length(self):
        window = list(range(30))
        feats = self.predictor._features(window)
        # 30 raw values + mean + std + min + max + slope + rate-of-change
        assert len(feats) == 30 + 6

    def test_mean_and_std_are_correct(self):
        window = [1.0, 2.0, 3.0, 4.0, 5.0]
        feats = self.predictor._features(window)
        arr = np.array(window)
        assert feats[len(window)] == pytest.approx(float(np.mean(arr)))
        assert feats[len(window) + 1] == pytest.approx(float(np.std(arr)))

    def test_min_and_max_are_correct(self):
        window = [5.0, 1.0, 9.0, 3.0]
        feats = self.predictor._features(window)
        n = len(window)
        assert feats[n + 2] == pytest.approx(1.0)
        assert feats[n + 3] == pytest.approx(9.0)

    def test_slope_is_zero_for_flat_window(self):
        window = [4.0] * 10
        feats = self.predictor._features(window)
        slope = feats[len(window) + 4]
        assert slope == pytest.approx(0.0, abs=1e-9)

    def test_slope_is_positive_for_increasing_window(self):
        window = list(np.arange(10, dtype=float))
        feats = self.predictor._features(window)
        slope = feats[len(window) + 4]
        assert slope > 0

    def test_slope_is_negative_for_decreasing_window(self):
        window = list(np.arange(10, 0, -1, dtype=float))
        feats = self.predictor._features(window)
        slope = feats[len(window) + 4]
        assert slope < 0

    def test_rate_of_change_needs_five_points(self):
        window = [1.0, 2.0, 3.0]
        feats = self.predictor._features(window)
        assert feats[-1] == pytest.approx(0.0)

    def test_rate_of_change_uses_last_minus_fifth_from_last(self):
        window = [0.0, 0.0, 0.0, 0.0, 0.0, 10.0]
        feats = self.predictor._features(window)
        assert feats[-1] == pytest.approx(10.0 - 0.0)

    def test_single_value_window_has_zero_slope(self):
        feats = self.predictor._features([5.0])
        n = 1
        assert feats[n + 4] == pytest.approx(0.0)


class TestDensityPredictorLevels:
    def setup_method(self):
        self.predictor = TrafficDensityPredictor()

    def test_below_low_threshold_is_low(self):
        assert self.predictor._to_level(config.DENSITY_LOW - 0.01) == "LOW"

    def test_at_low_threshold_is_medium(self):
        # _to_level uses strict '<', so hitting the threshold exactly rolls
        # up to the next band.
        assert self.predictor._to_level(config.DENSITY_LOW) == "MEDIUM"

    def test_between_thresholds_is_medium(self):
        mid = (config.DENSITY_LOW + config.DENSITY_MEDIUM) / 2
        assert self.predictor._to_level(mid) == "MEDIUM"

    def test_at_medium_threshold_is_high(self):
        assert self.predictor._to_level(config.DENSITY_MEDIUM) == "HIGH"

    def test_well_above_medium_is_high(self):
        assert self.predictor._to_level(config.DENSITY_MEDIUM * 10) == "HIGH"


class TestDensityPredictorUpdateAndPredict:
    def test_predict_with_no_history_returns_zero(self):
        predictor = TrafficDensityPredictor()
        density, level, confidence = predictor.predict()
        assert density == 0.0
        assert level == "LOW"
        assert confidence == 0.0

    def test_predict_before_full_window_uses_mean(self):
        predictor = TrafficDensityPredictor(window_size=30, predict_horizon=15)
        for v in [2.0, 4.0, 6.0]:
            predictor.update(v)
        density, _, confidence = predictor.predict()
        assert density == pytest.approx(4.0)
        assert confidence == 0.0

    def test_predict_after_full_window_uses_ema_when_untrained(self):
        predictor = TrafficDensityPredictor(window_size=5, predict_horizon=2, min_samples=10_000)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            predictor.update(v)
        density, level, confidence = predictor.predict()
        assert not predictor.is_trained
        assert confidence == 0.0
        assert density > 0

    def test_history_does_not_grow_unbounded(self):
        predictor = TrafficDensityPredictor(window_size=10, predict_horizon=5)
        for i in range(10_000):
            predictor.update(float(i % 20))
        assert len(predictor.history) <= predictor.history.maxlen

    def test_training_pairs_accumulate_once_enough_history(self):
        predictor = TrafficDensityPredictor(window_size=5, predict_horizon=3, min_samples=100_000)
        for i in range(50):
            predictor.update(float(i))
        assert len(predictor.X_train) > 0
        assert len(predictor.X_train) == len(predictor.y_train)

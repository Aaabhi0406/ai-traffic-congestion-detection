"""Tests for detection.py's filtering logic and config.VEHICLE_WEIGHTS,
using a stub YOLO model so no real ultralytics/torch dependency or model
weights file is needed."""

import pytest

from traffic_system import config
from traffic_system.detection import detect_vehicles


class _FakeTensor:
    """Minimal stand-in for the torch tensors ultralytics returns, just
    enough to support indexing and float()/int() conversion."""

    def __init__(self, values):
        self._values = list(values)

    def __getitem__(self, idx):
        return self._values[idx]


class _FakeBox:
    def __init__(self, cls_id, conf, xyxy):
        self.cls = _FakeTensor([cls_id])
        self.conf = _FakeTensor([conf])
        self.xyxy = _FakeTensor([xyxy])


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    """Callable stub mimicking `model(frame, conf=..., iou=..., imgsz=..., verbose=...)[0]`."""

    names = config.CLASS_NAMES

    def __init__(self, boxes):
        self._boxes = boxes

    def __call__(self, frame, conf, iou, imgsz, verbose):
        return [_FakeResult(self._boxes)]


FRAME_W, FRAME_H = 1920, 1080


def make_box(cls_id, conf, x1=100, y1=100, x2=200, y2=200):
    return _FakeBox(cls_id, conf, [x1, y1, x2, y2])


class TestDetectVehiclesClassFilter:
    def test_non_vehicle_class_is_dropped(self):
        # class 0 = person in COCO, not in VEHICLE_CLASSES
        model = _FakeModel([make_box(0, 0.99)])
        result = detect_vehicles(None, model, FRAME_W, FRAME_H)
        assert result == []

    def test_car_above_threshold_is_kept(self):
        model = _FakeModel([make_box(2, config.CONF_PER_CLASS[2] + 0.1)])
        result = detect_vehicles(None, model, FRAME_W, FRAME_H)
        assert len(result) == 1
        assert result[0].class_name == "car"


class TestDetectVehiclesConfidenceFilter:
    def test_car_below_threshold_is_dropped(self):
        model = _FakeModel([make_box(2, config.CONF_PER_CLASS[2] - 0.05)])
        result = detect_vehicles(None, model, FRAME_W, FRAME_H)
        assert result == []

    def test_motorcycle_uses_its_own_lower_threshold(self):
        # A confidence that would fail the car threshold should still pass
        # for motorcycle, since motorcycles have a lower bar.
        conf = (config.CONF_PER_CLASS[3] + config.CONF_PER_CLASS[2]) / 2
        assert conf < config.CONF_PER_CLASS[2]
        assert conf >= config.CONF_PER_CLASS[3]
        model = _FakeModel([make_box(3, conf)])
        result = detect_vehicles(None, model, FRAME_W, FRAME_H)
        assert len(result) == 1

    def test_min_conf_override_raises_the_bar(self):
        conf = config.CONF_PER_CLASS[3] + 0.02  # passes tuned motorcycle threshold
        model = _FakeModel([make_box(3, conf)])

        result_default = detect_vehicles(None, model, FRAME_W, FRAME_H)
        assert len(result_default) == 1

        result_overridden = detect_vehicles(
            None, model, FRAME_W, FRAME_H, min_conf_override=0.95
        )
        assert result_overridden == []

    def test_min_conf_override_cannot_lower_the_bar(self):
        # A car just below its tuned threshold should still be rejected even
        # if the override is set lower than that threshold.
        conf = config.CONF_PER_CLASS[2] - 0.05
        model = _FakeModel([make_box(2, conf)])
        result = detect_vehicles(None, model, FRAME_W, FRAME_H, min_conf_override=0.1)
        assert result == []


class TestDetectVehiclesSizeFilter:
    def test_box_too_small_is_dropped(self):
        model = _FakeModel([make_box(2, 0.9, x1=0, y1=0, x2=5, y2=5)])
        result = detect_vehicles(None, model, FRAME_W, FRAME_H)
        assert result == []

    def test_box_too_large_is_dropped(self):
        model = _FakeModel([make_box(2, 0.9, x1=0, y1=0, x2=FRAME_W, y2=FRAME_H)])
        result = detect_vehicles(None, model, FRAME_W, FRAME_H)
        assert result == []

    def test_normal_sized_box_is_kept(self):
        model = _FakeModel([make_box(2, 0.9, x1=100, y1=100, x2=250, y2=220)])
        result = detect_vehicles(None, model, FRAME_W, FRAME_H)
        assert len(result) == 1


class TestDetectionHelpers:
    def test_centre_is_box_midpoint(self):
        model = _FakeModel([make_box(2, 0.9, x1=100, y1=100, x2=200, y2=300)])
        det = detect_vehicles(None, model, FRAME_W, FRAME_H)[0]
        assert det.centre == (150, 200)

    def test_deepsort_input_format(self):
        model = _FakeModel([make_box(2, 0.9, x1=100, y1=100, x2=200, y2=300)])
        det = detect_vehicles(None, model, FRAME_W, FRAME_H)[0]
        bbox, conf, cls_id = det.as_deepsort_input()
        assert bbox == [100, 100, 100, 200]  # [x, y, w, h]
        assert cls_id == 2


class TestVehicleWeights:
    @pytest.mark.parametrize("name,expected", [
        ("bicycle", 0.3), ("car", 1.0), ("motorcycle", 0.5),
        ("bus", 2.0), ("truck", 2.0),
    ])
    def test_known_weights(self, name, expected):
        assert config.VEHICLE_WEIGHTS[name] == pytest.approx(expected)

    def test_bus_and_truck_weigh_more_than_car(self):
        assert config.VEHICLE_WEIGHTS["bus"] > config.VEHICLE_WEIGHTS["car"]
        assert config.VEHICLE_WEIGHTS["truck"] > config.VEHICLE_WEIGHTS["car"]

    def test_bicycle_weighs_least(self):
        assert config.VEHICLE_WEIGHTS["bicycle"] == min(config.VEHICLE_WEIGHTS.values())

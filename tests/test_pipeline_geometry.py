"""Tests for the pure-geometry pieces of pipeline.py: ROI construction and
the centre-in-polygon test. These need OpenCV/NumPy but not YOLO or
DeepSort, so they run in plain CI without a model download."""


from traffic_system.pipeline import build_roi, centre_in_roi


class TestBuildRoi:
    def test_returns_four_points(self):
        roi = build_roi(frame_h=1080, frame_w=1920)
        assert roi.shape == (4, 2)

    def test_top_edge_is_inset_from_frame_edges(self):
        roi = build_roi(frame_h=1080, frame_w=1920)
        top_l_x, top_r_x = roi[0][0], roi[1][0]
        assert top_l_x > 0
        assert top_r_x < 1920

    def test_bottom_edge_spans_full_width(self):
        roi = build_roi(frame_h=1080, frame_w=1920)
        bottom_r_x, bottom_l_x = roi[2][0], roi[3][0]
        assert bottom_r_x == 1920
        assert bottom_l_x == 0

    def test_scales_with_frame_size(self):
        small = build_roi(frame_h=480, frame_w=640)
        large = build_roi(frame_h=1080, frame_w=1920)
        assert large[1][0] > small[1][0]  # top-right x scales up


class TestCentreInRoi:
    def setup_method(self):
        self.roi = build_roi(frame_h=1080, frame_w=1920)

    def test_frame_centre_is_inside(self):
        assert centre_in_roi(960, 540, self.roi) is True

    def test_far_outside_frame_is_outside(self):
        assert centre_in_roi(-500, -500, self.roi) is False

    def test_top_left_corner_area_outside_inset(self):
        # Just inside the frame's literal top-left corner, but the ROI's top
        # edge is inset — a point right at (2, 2) should fall outside the
        # trapezoid since the top-left ROI vertex is at ~2.5% of width.
        assert centre_in_roi(2, 2, self.roi) is False

    def test_near_bottom_of_frame_is_inside(self):
        # ROI bottom edge sits at 99% of frame height (~1069 on a 1080px
        # frame), not the literal last row, so check just above that.
        assert centre_in_roi(960, 1060, self.roi) is True

    def test_point_on_or_near_boundary_does_not_crash(self):
        # pointPolygonTest is well-defined on the boundary (returns 0);
        # just confirm it doesn't raise for edge coordinates.
        result = centre_in_roi(0, 1078, self.roi)
        assert isinstance(result, bool)

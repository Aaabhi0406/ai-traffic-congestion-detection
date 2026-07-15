"""Tests for tracking.py's pure-logic helpers: line-crossing detection and
ghost-track suppression. Neither needs a real DeepSort instance."""

from traffic_system.tracking import LineCrossingCounter, has_nearby_detection


class TestLineCrossingCounter:
    def test_no_crossing_when_staying_above_line(self):
        counter = LineCrossingCounter()
        line_y = 500

        crossed = counter.update(track_id=1, centre_y=100, line_y=line_y)
        counter.commit_frame({1: 100})
        assert crossed is False
        assert counter.total == 0

        crossed = counter.update(track_id=1, centre_y=200, line_y=line_y)
        counter.commit_frame({1: 200})
        assert crossed is False
        assert counter.total == 0

    def test_crossing_top_to_bottom_is_counted(self):
        counter = LineCrossingCounter()
        line_y = 500

        counter.update(track_id=1, centre_y=480, line_y=line_y)
        counter.commit_frame({1: 480})

        crossed = counter.update(track_id=1, centre_y=520, line_y=line_y)
        counter.commit_frame({1: 520})

        assert crossed is True
        assert counter.total == 1

    def test_crossing_bottom_to_top_is_not_counted(self):
        """Only top->bottom crossings count (matches the throughput-direction
        assumption baked into the original implementation)."""
        counter = LineCrossingCounter()
        line_y = 500

        counter.update(track_id=1, centre_y=520, line_y=line_y)
        counter.commit_frame({1: 520})

        crossed = counter.update(track_id=1, centre_y=480, line_y=line_y)
        counter.commit_frame({1: 480})

        assert crossed is False
        assert counter.total == 0

    def test_new_track_id_with_no_history_does_not_crash(self):
        counter = LineCrossingCounter()
        crossed = counter.update(track_id=99, centre_y=600, line_y=500)
        assert crossed is False

    def test_multiple_tracks_counted_independently(self):
        counter = LineCrossingCounter()
        line_y = 500

        counter.update(track_id=1, centre_y=480, line_y=line_y)
        counter.update(track_id=2, centre_y=480, line_y=line_y)
        counter.commit_frame({1: 480, 2: 480})

        counter.update(track_id=1, centre_y=520, line_y=line_y)
        counter.update(track_id=2, centre_y=490, line_y=line_y)  # stays above
        counter.commit_frame({1: 520, 2: 490})

        assert counter.total == 1

    def test_reset_clears_state(self):
        counter = LineCrossingCounter()
        counter.update(track_id=1, centre_y=480, line_y=500)
        counter.commit_frame({1: 480})
        counter.update(track_id=1, centre_y=520, line_y=500)
        counter.commit_frame({1: 520})
        assert counter.total == 1

        counter.reset()
        assert counter.total == 0
        # After reset, track 1 has no prior centre, so no crossing is
        # detected even if this frame's position is below the line.
        crossed = counter.update(track_id=1, centre_y=520, line_y=500)
        assert crossed is False


class TestHasNearbyDetection:
    def test_true_when_centre_inside_box(self):
        assert has_nearby_detection(0, 0, 100, 100, [(50, 50)]) is True

    def test_true_when_centre_within_threshold_outside_box(self):
        assert has_nearby_detection(0, 0, 100, 100, [(150, 50)], threshold=60) is True

    def test_false_when_centre_far_outside_box(self):
        assert has_nearby_detection(0, 0, 100, 100, [(500, 500)], threshold=60) is False

    def test_false_with_no_detections(self):
        assert has_nearby_detection(0, 0, 100, 100, []) is False

    def test_true_if_any_of_multiple_centres_matches(self):
        centres = [(9999, 9999), (50, 50)]
        assert has_nearby_detection(0, 0, 100, 100, centres) is True

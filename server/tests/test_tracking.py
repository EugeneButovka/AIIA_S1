from tracking import TrackRegistry

PERSON = 0
CAR = 2


def test_track_counts_after_two_consecutive_frames():
    registry = TrackRegistry(2)
    registry.observe_frame([(1, PERSON)])
    assert registry.unique_persons() == 0
    registry.observe_frame([(1, PERSON)])
    assert registry.unique_persons() == 1


def test_streak_resets_when_track_disappears():
    registry = TrackRegistry(2)
    registry.observe_frame([(1, PERSON)])
    registry.observe_frame([])
    registry.observe_frame([(1, PERSON)])
    assert registry.unique_persons() == 0


def test_confirmed_track_stays_unique_after_gap():
    registry = TrackRegistry(2)
    registry.observe_frame([(1, PERSON)])
    registry.observe_frame([(1, PERSON)])
    registry.observe_frame([])
    registry.observe_frame([(1, PERSON)])
    assert registry.unique_persons() == 1


def test_single_frame_gate_counts_immediately():
    registry = TrackRegistry(1)
    registry.observe_frame([(7, CAR)])
    assert registry.unique_vehicles() == 1


def test_persons_and_vehicles_counted_separately():
    registry = TrackRegistry(1)
    registry.observe_frame([(1, PERSON), (2, CAR)])
    assert registry.unique_persons() == 1
    assert registry.unique_vehicles() == 1


def test_unrelated_class_is_ignored():
    registry = TrackRegistry(1)
    registry.observe_frame([(1, 15)])
    assert registry.unique_persons() == 0
    assert registry.unique_vehicles() == 0


def test_reset_clears_registry():
    registry = TrackRegistry(1)
    registry.observe_frame([(1, PERSON)])
    registry.reset()
    assert registry.unique_persons() == 0
    registry.observe_frame([(1, PERSON)])
    assert registry.unique_persons() == 1
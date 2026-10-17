from storage import COLUMNS, PredictionRecord, ResultsStore


def make_record():
    return PredictionRecord(
        timestamp="2026-10-17T00:00:00+00:00",
        persons=1,
        detections=1,
        avg_confidence=0.5,
        inference_ms=10.0,
        response_ms=20.0,
        image_bytes=100,
        response_bytes=200,
        unique_persons=2,
        unique_vehicles=1,
    )


def test_append_writes_current_schema(tmp_path):
    store = ResultsStore(tmp_path / "results.csv")
    store.append(make_record())
    lines = (tmp_path / "results.csv").read_text().splitlines()
    assert lines[0].split(",") == COLUMNS
    assert lines[1].split(",")[-2:] == ["2", "1"]


def test_legacy_rows_write_empty_unique_columns(tmp_path):
    record = PredictionRecord(
        timestamp="2026-10-17T00:00:00+00:00",
        persons=1,
        detections=1,
        avg_confidence=0.5,
        inference_ms=10.0,
        response_ms=20.0,
        image_bytes=100,
        response_bytes=200,
    )
    store = ResultsStore(tmp_path / "results.csv")
    store.append(record)
    row = (tmp_path / "results.csv").read_text().splitlines()[1].split(",")
    assert row[-2:] == ["", ""]
    assert len(row) == len(COLUMNS)


def test_old_schema_file_is_rotated(tmp_path):
    old_path = tmp_path / "results.csv"
    old_path.write_text(
        "timestamp_utc,persons,detections,avg_confidence,inference_ms,response_ms,image_bytes,response_bytes\n"
        "2026-09-30,1,1,0.5,10,20,100,200\n"
    )
    store = ResultsStore(old_path)
    store.append(make_record())
    rotated = list(tmp_path.glob("results_*.csv"))
    assert len(rotated) == 1
    assert old_path.exists()
    lines = old_path.read_text().splitlines()
    assert lines[0].split(",") == COLUMNS


def test_same_schema_file_is_appended_without_rotation(tmp_path):
    path = tmp_path / "results.csv"
    store = ResultsStore(path)
    store.append(make_record())
    store.append(make_record())
    assert list(tmp_path.glob("results_*.csv")) == []
    assert len(path.read_text().splitlines()) == 3
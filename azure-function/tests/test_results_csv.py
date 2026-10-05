import csv
import io

import pytest
from azure.core.exceptions import ResourceExistsError

from results import (
    AnalysisRecord,
    append_analysis_row,
    build_blob_service,
    build_csv_header,
    build_csv_row,
    upload_processed_image,
)


def build_record(**overrides):
    defaults = dict(
        timestamp_utc="2026-10-01T10:00:00+00:00",
        persons=2,
        detections=4,
        avg_confidence=0.90758,
        inference_ms=150.456,
        response_ms=200.789,
        image_bytes=1024,
        response_bytes=512,
        blob_name="uploads/frame 1.jpg",
        caption="a person, pointing at a screen",
        caption_confidence=0.48921,
        tags="person:0.9812",
        ocr_text='He said "hello"\nnext line',
    )
    return AnalysisRecord(**{**defaults, **overrides})


class FakeAppendBlob:
    def __init__(self, state, container_name, blob_name):
        self.state = state
        self.key = (container_name, blob_name)

    def exists(self):
        return self.key in self.state["blobs"]

    def create(self, content_settings):
        if self.key in self.state["blobs"]:
            raise ResourceExistsError(message="blob already exists")
        self.state["blobs"][self.key] = b""
        self.state["settings"][self.key] = content_settings

    def append_block(self, data):
        self.state["blobs"][self.key] += data


class FakeBlockBlob:
    def __init__(self, state, container_name, blob_name):
        self.state = state
        self.key = (container_name, blob_name)

    def upload_blob(self, data, overwrite=False, content_settings=None):
        self.state["images"][self.key] = data
        self.state["image_uploads"].append(
            {"key": self.key, "overwrite": overwrite, "content_settings": content_settings}
        )


class FakeContainer:
    def __init__(self, state, container_name):
        self.state = state
        self.container_name = container_name

    def exists(self):
        return self.container_name in self.state["containers"]

    def create_container(self):
        if self.container_name in self.state["containers"]:
            raise ResourceExistsError(message="container already exists")
        self.state["containers"].add(self.container_name)

    def get_append_blob_client(self, blob_name):
        return FakeAppendBlob(self.state, self.container_name, blob_name)

    def get_blob_client(self, blob_name):
        return FakeBlockBlob(self.state, self.container_name, blob_name)


class FakeBlobServiceClient:
    def __init__(self, state):
        self.state = state

    def get_container_client(self, container_name):
        return FakeContainer(self.state, container_name)


def install_fake_storage():
    state = {"containers": set(), "blobs": {}, "settings": {}, "images": {}, "image_uploads": []}
    return state, FakeBlobServiceClient(state)


def test_build_blob_service_uses_connection_string(monkeypatch):
    # given
    monkeypatch.setenv("AzureWebJobsStorage", "fake-connection-string")
    captured = {}

    def fake_from_connection_string(connection_string):
        captured["connection_string"] = connection_string
        return "fake-service"

    monkeypatch.setattr(
        "results.BlobServiceClient.from_connection_string",
        staticmethod(fake_from_connection_string),
    )

    # when
    service = build_blob_service()

    # then
    assert service == "fake-service"
    assert captured["connection_string"] == "fake-connection-string"


def test_build_blob_service_uses_managed_identity(monkeypatch):
    # given
    monkeypatch.delenv("AzureWebJobsStorage", raising=False)
    monkeypatch.setenv("AzureWebJobsStorage__blobServiceUri", "https://account.blob.core.windows.net")
    monkeypatch.setenv("AzureWebJobsStorage__credential", "managedIdentity")
    monkeypatch.setenv("AzureWebJobsStorage__clientId", "client-123")
    created = {}

    class FakeCredential:
        def __init__(self, **kwargs):
            created["credential_kwargs"] = kwargs

    class FakeBlobServiceClientConstructor:
        def __init__(self, account_url=None, credential=None):
            created["account_url"] = account_url
            created["credential"] = credential

    monkeypatch.setattr("results.DefaultAzureCredential", FakeCredential)
    monkeypatch.setattr("results.BlobServiceClient", FakeBlobServiceClientConstructor)

    # when
    build_blob_service()

    # then
    assert created["credential_kwargs"] == {"managed_identity_client_id": "client-123"}
    assert created["account_url"] == "https://account.blob.core.windows.net"
    assert isinstance(created["credential"], FakeCredential)


def test_build_blob_service_uses_default_credential_without_client_id(monkeypatch):
    # given
    monkeypatch.delenv("AzureWebJobsStorage", raising=False)
    monkeypatch.setenv("AzureWebJobsStorage__blobServiceUri", "https://account.blob.core.windows.net")
    monkeypatch.delenv("AzureWebJobsStorage__clientId", raising=False)
    created = {}

    class FakeCredential:
        def __init__(self, **kwargs):
            created["credential_kwargs"] = kwargs

    monkeypatch.setattr("results.DefaultAzureCredential", FakeCredential)
    monkeypatch.setattr("results.BlobServiceClient", lambda **kwargs: kwargs)

    # when
    build_blob_service()

    # then
    assert created["credential_kwargs"] == {}


def test_build_blob_service_requires_storage_configuration(monkeypatch):
    # given
    monkeypatch.delenv("AzureWebJobsStorage", raising=False)
    monkeypatch.delenv("AzureWebJobsStorage__blobServiceUri", raising=False)

    # when / then
    with pytest.raises(RuntimeError):
        build_blob_service()


def test_build_csv_header_lists_all_columns():
    # given
    # when
    header = build_csv_header()

    # then
    assert header == (
        b"timestamp_utc,persons,detections,avg_confidence,inference_ms,response_ms,"
        b"image_bytes,response_bytes,blob_name,caption,caption_confidence,tags,ocr_text\r\n"
    )


def test_build_csv_row_escapes_special_characters():
    # given
    record = build_record()

    # when
    row = build_csv_row(record)

    # then
    parsed = list(csv.reader(io.StringIO(row.decode("utf-8"))))[0]
    assert parsed == [
        "2026-10-01T10:00:00+00:00",
        "2",
        "4",
        "0.9076",
        "150.5",
        "200.8",
        "1024",
        "512",
        "uploads/frame 1.jpg",
        "a person, pointing at a screen",
        "0.4892",
        "person:0.9812",
        'He said "hello"\nnext line',
    ]


def test_append_analysis_row_creates_container_blob_and_header():
    # given
    state, service = install_fake_storage()
    record = build_record(blob_name="uploads/frame.jpg", ocr_text="hello")

    # when
    append_analysis_row(service, "results", "analysis.csv", record)

    # then
    assert "results" in state["containers"]
    content = state["blobs"][("results", "analysis.csv")]
    assert content.startswith(build_csv_header())
    assert content.count(b"\r\n") == 2
    assert state["settings"][("results", "analysis.csv")].content_type == "text/csv"


def test_append_analysis_row_appends_without_duplicating_header():
    # given
    state, service = install_fake_storage()
    append_analysis_row(service, "results", "analysis.csv", build_record(ocr_text="hello"))

    # when
    append_analysis_row(service, "results", "analysis.csv", build_record(ocr_text="hello"))

    # then
    content = state["blobs"][("results", "analysis.csv")]
    assert content.count(build_csv_header()) == 1
    assert content.count(b"\r\n") == 3


def test_append_analysis_row_uses_existing_container():
    # given
    state, service = install_fake_storage()
    state["containers"].add("results")

    # when
    append_analysis_row(service, "results", "analysis.csv", build_record(ocr_text="hello"))

    # then
    assert "results" in state["containers"]
    assert ("results", "analysis.csv") in state["blobs"]


def test_upload_processed_image_creates_container_and_uploads_jpeg():
    # given
    state, service = install_fake_storage()

    # when
    upload_processed_image(service, "results", "processed_frame.jpg", b"jpeg-bytes")

    # then
    assert "results" in state["containers"]
    assert state["images"][("results", "processed_frame.jpg")] == b"jpeg-bytes"
    upload = state["image_uploads"][0]
    assert upload["overwrite"] is True
    assert upload["content_settings"].content_type == "image/jpeg"


def test_upload_processed_image_overwrites_existing_blob():
    # given
    state, service = install_fake_storage()
    state["containers"].add("results")
    state["images"][("results", "processed_frame.jpg")] = b"old-bytes"

    # when
    upload_processed_image(service, "results", "processed_frame.jpg", b"new-bytes")

    # then
    assert state["images"][("results", "processed_frame.jpg")] == b"new-bytes"
    assert len(state["image_uploads"]) == 1
    assert state["image_uploads"][0]["overwrite"] is True
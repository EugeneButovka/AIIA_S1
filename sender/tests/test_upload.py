from datetime import datetime

import cv2
import numpy as np
from azure.core.exceptions import ResourceExistsError

import sender
from sender import AZURE_UPLOAD_CONTAINER, build_blob_name, build_uploads_container, upload_frame


def build_frame(width=40, height=30):
    return np.zeros((height, width, 3), dtype=np.uint8)


class FakeBlobServiceClient:
    def __init__(self, container):
        self.container = container
        self.requested_container = None

    def get_container_client(self, container_name):
        self.requested_container = container_name
        return self.container


class FakeContainerClient:
    def __init__(self, exists=True):
        self.exists_value = exists
        self.created = False
        self.uploads = []

    def exists(self):
        return self.exists_value

    def create_container(self):
        if self.exists_value:
            raise ResourceExistsError(message="container already exists")
        self.created = True

    def upload_blob(self, name, data, content_settings=None):
        self.uploads.append({"name": name, "data": data, "content_settings": content_settings})


def install_fake_storage(monkeypatch, container):
    monkeypatch.setattr(sender, "AZURE_STORAGE_CONNECTION_STRING", "fake-connection-string")
    monkeypatch.setattr(sender, "AZURE_UPLOAD_CONTAINER", "uploads")
    monkeypatch.setattr(
        "sender.BlobServiceClient.from_connection_string",
        staticmethod(lambda connection_string: FakeBlobServiceClient(container)),
    )
    return container


def test_build_blob_name_is_timestamped_jpeg():
    # given
    # when
    blob_name = build_blob_name(datetime(2026, 10, 5, 23, 30, 15, 123456))

    # then
    assert blob_name == "frame_20261005_233015_123456.jpg"


def test_upload_frame_uploads_encoded_jpeg():
    # given
    container = FakeContainerClient()

    # when
    blob_name = upload_frame(container, build_frame())

    # then
    assert blob_name is not None
    upload = container.uploads[0]
    assert upload["name"] == blob_name
    assert upload["content_settings"].content_type == "image/jpeg"
    decoded = cv2.imdecode(np.frombuffer(upload["data"], np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == (30, 40, 3)


def test_upload_frame_returns_none_for_empty_frame():
    # given
    container = FakeContainerClient()

    # when
    blob_name = upload_frame(container, np.zeros((0, 0, 3), dtype=np.uint8))

    # then
    assert blob_name is None
    assert container.uploads == []


def test_build_uploads_container_creates_missing_container(monkeypatch):
    # given
    container = FakeContainerClient(exists=False)
    install_fake_storage(monkeypatch, container)

    # when
    result = build_uploads_container()

    # then
    assert result is container
    assert container.created is True


def test_build_uploads_container_reuses_existing_container(monkeypatch):
    # given
    container = FakeContainerClient(exists=True)
    install_fake_storage(monkeypatch, container)

    # when
    result = build_uploads_container()

    # then
    assert result is container
    assert container.created is False
import csv
import io
import os
from dataclasses import dataclass

from azure.core.exceptions import ResourceExistsError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings

CSV_COLUMNS: tuple[str, ...] = (
    "timestamp_utc",
    "persons",
    "detections",
    "avg_confidence",
    "inference_ms",
    "response_ms",
    "image_bytes",
    "response_bytes",
    "blob_name",
    "caption",
    "caption_confidence",
    "tags",
    "ocr_text",
)
DEFAULT_RESULTS_CONTAINER = "results"
DEFAULT_RESULTS_CSV = "analysis.csv"


@dataclass(frozen=True)
class AnalysisRecord:
    timestamp_utc: str
    persons: int
    detections: int
    avg_confidence: float
    inference_ms: float
    response_ms: float
    image_bytes: int
    response_bytes: int
    blob_name: str
    caption: str
    caption_confidence: float
    tags: str
    ocr_text: str

    def to_row(self) -> list:
        return [
            self.timestamp_utc,
            self.persons,
            self.detections,
            round(self.avg_confidence, 4),
            round(self.inference_ms, 1),
            round(self.response_ms, 1),
            self.image_bytes,
            self.response_bytes,
            self.blob_name,
            self.caption,
            round(self.caption_confidence, 4),
            self.tags,
            self.ocr_text,
        ]


def build_csv_header() -> bytes:
    return (",".join(CSV_COLUMNS) + "\r\n").encode("utf-8")


def build_csv_row(record: AnalysisRecord) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(record.to_row())
    return buffer.getvalue().encode("utf-8")


def build_blob_service() -> BlobServiceClient:
    connection_string = os.environ.get("AzureWebJobsStorage")
    if connection_string:
        return BlobServiceClient.from_connection_string(connection_string)
    blob_service_uri = os.environ.get("AzureWebJobsStorage__blobServiceUri")
    if not blob_service_uri:
        raise RuntimeError("AzureWebJobsStorage or AzureWebJobsStorage__blobServiceUri is required")
    client_id = os.environ.get("AzureWebJobsStorage__clientId")
    if client_id:
        credential = DefaultAzureCredential(managed_identity_client_id=client_id)
    else:
        credential = DefaultAzureCredential()
    return BlobServiceClient(account_url=blob_service_uri, credential=credential)


def append_analysis_row(
    service: BlobServiceClient,
    container_name: str,
    csv_name: str,
    record: AnalysisRecord,
) -> None:
    container = service.get_container_client(container_name)
    if not container.exists():
        try:
            container.create_container()
        except ResourceExistsError:
            pass
    append_blob = container.get_append_blob_client(csv_name)
    if not append_blob.exists():
        try:
            append_blob.create(content_settings=ContentSettings(content_type="text/csv"))
        except ResourceExistsError:
            pass
        else:
            append_blob.append_block(build_csv_header())
    append_blob.append_block(build_csv_row(record))


def upload_processed_image(
    service: BlobServiceClient,
    container_name: str,
    blob_name: str,
    image_data: bytes,
) -> None:
    container = service.get_container_client(container_name)
    if not container.exists():
        try:
            container.create_container()
        except ResourceExistsError:
            pass
    container.get_blob_client(blob_name).upload_blob(
        image_data,
        overwrite=True,
        content_settings=ContentSettings(content_type="image/jpeg"),
    )
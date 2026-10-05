import json
import logging
import os
import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import Optional

# Azure Vision (Image Analysis 4.0) — disabled in favour of the local YOLO model:
# from azure.ai.vision.imageanalysis import ImageAnalysisClient
# from azure.ai.vision.imageanalysis.models import VisualFeatures
# from azure.core.credentials import AzureKeyCredential
# from azure.core.exceptions import HttpResponseError

from detection import Detector, DetectionResult, InvalidImageError
from processed_image import build_processed_blob_name, draw_detection_boxes
from results import (
    DEFAULT_RESULTS_CONTAINER,
    DEFAULT_RESULTS_CSV,
    AnalysisRecord,
    append_analysis_row,
    upload_processed_image,
)

logger = logging.getLogger(__name__)

DEFAULT_YOLO_WEIGHTS = "yolov8n.pt"

# VISUAL_FEATURES = [
#     VisualFeatures.CAPTION,
#     VisualFeatures.READ,
#     VisualFeatures.TAGS,
#     VisualFeatures.PEOPLE,
#     VisualFeatures.OBJECTS,
# ]

_detector: Optional[Detector] = None


def get_detector() -> Detector:
    global _detector
    if _detector is None:
        _detector = Detector(os.environ.get("YOLO_WEIGHTS") or DEFAULT_YOLO_WEIGHTS)
    return _detector


# def build_client() -> ImageAnalysisClient:
#     return ImageAnalysisClient(
#         endpoint=os.environ["VISION_ENDPOINT"],
#         credential=AzureKeyCredential(os.environ["VISION_KEY"]),
#     )


# def object_confidence(detected_object) -> float:
#     confidences = [tag.confidence for tag in (detected_object.tags or [])]
#     return max(confidences) if confidences else 0.0


# def extract_people_boxes(people) -> list:
#     return [person.bounding_box for person in people if person.confidence >= CONFIDENCE_THRESHOLD]


# def extract_object_boxes(objects) -> list:
#     return [
#         detected_object.bounding_box
#         for detected_object in objects
#         if object_confidence(detected_object) >= CONFIDENCE_THRESHOLD
#     ]


def build_response_payload(result: DetectionResult, response_ms: float) -> dict:
    return {
        "detections": result.detections,
        "persons": result.persons,
        "inference_ms": round(result.inference_ms, 1),
        "response_ms": round(response_ms, 1),
    }


# def build_response_payload(record: AnalysisRecord) -> dict:
#     return {
#         "blob_name": record.blob_name,
#         "persons": record.persons,
#         "detections": record.detections,
#         "avg_confidence": round(record.avg_confidence, 4),
#         "inference_ms": round(record.inference_ms, 1),
#         "response_ms": round(record.response_ms, 1),
#         "caption": record.caption,
#         "caption_confidence": round(record.caption_confidence, 4),
#         "tags": record.tags,
#         "ocr_text": record.ocr_text,
#     }


def to_analysis_record(
    result: DetectionResult,
    blob_name: str,
    image_data: bytes,
    response_ms: float,
) -> AnalysisRecord:
    payload = build_response_payload(result, response_ms)
    record = AnalysisRecord(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        persons=result.persons,
        detections=len(result.detections),
        avg_confidence=result.avg_confidence,
        inference_ms=result.inference_ms,
        response_ms=response_ms,
        image_bytes=len(image_data),
        response_bytes=0,
        blob_name=blob_name,
        caption="",
        caption_confidence=0.0,
        tags="",
        ocr_text="",
    )
    return replace(record, response_bytes=len(json.dumps(payload).encode("utf-8")))


def analyze_blob(blob_name: str, image_data: bytes) -> None:
    started_at = time.perf_counter()
    # client = build_client()
    # analyze_started_at = time.perf_counter()
    # try:
    #     result = client.analyze(
    #         image_data=image_data,
    #         visual_features=VISUAL_FEATURES,
    #         gender_neutral_caption=True,
    #     )
    # except HttpResponseError as error:
    #     logger.warning("skipped %s: %s", blob_name, error.message)
    #     return
    # inference_ms = (time.perf_counter() - analyze_started_at) * 1000
    try:
        result = get_detector().detect(image_data)
    except InvalidImageError as error:
        logger.warning("skipped %s: %s", blob_name, error)
        return
    response_ms = (time.perf_counter() - started_at) * 1000
    record = to_analysis_record(result, blob_name, image_data, response_ms)
    connection_string = os.environ["AzureWebJobsStorage"]
    container_name = os.environ.get("RESULTS_CONTAINER_NAME", DEFAULT_RESULTS_CONTAINER)
    csv_name = os.environ.get("RESULTS_CSV_NAME", DEFAULT_RESULTS_CSV)
    append_analysis_row(connection_string, container_name, csv_name, record)
    # people = result.people.list if result.people else []
    # objects = result.objects.list if result.objects else []
    # processed_image = draw_detection_boxes(image_data, extract_people_boxes(people), extract_object_boxes(objects))
    processed_image = draw_detection_boxes(image_data, result.detections)
    if processed_image is None:
        logger.warning("could not render processed image for %s", blob_name)
        return
    upload_processed_image(connection_string, container_name, build_processed_blob_name(blob_name), processed_image)
    logger.info("analyzed %s", blob_name)
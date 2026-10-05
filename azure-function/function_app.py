import json
import time

import azure.functions as func

from analysis import analyze_blob, build_response_payload, get_detector, process_prediction
from detection import InvalidImageError
from processed_image import build_request_blob_name

app = func.FunctionApp()


@app.route(route="hello", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def hello(req: func.HttpRequest) -> func.HttpResponse:
    name = req.params.get("name")
    if not name:
        try:
            name = req.get_json().get("name")
        except (ValueError, AttributeError):
            pass
    if name:
        return func.HttpResponse(f"Hello, {name}!", status_code=200)
    return func.HttpResponse(
        "Pass a name via the 'name' query parameter or JSON body",
        status_code=400,
    )


def read_image_data(req: func.HttpRequest):
    file = req.files.get("file")
    if file is not None:
        return file.read()
    return req.get_body() or None


@app.route(route="predict", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
def predict(req: func.HttpRequest) -> func.HttpResponse:
    started_at = time.perf_counter()
    image_data = read_image_data(req)
    if not image_data:
        return func.HttpResponse(
            "Missing image file: POST multipart/form-data with a 'file' field or a raw image body",
            status_code=400,
        )
    try:
        result = get_detector().detect(image_data)
    except InvalidImageError:
        return func.HttpResponse("Invalid or unreadable image file", status_code=400)
    response_ms = (time.perf_counter() - started_at) * 1000
    process_prediction(build_request_blob_name(), image_data, result, response_ms)
    payload = build_response_payload(result, response_ms)
    return func.HttpResponse(json.dumps(payload), status_code=200, mimetype="application/json")


@app.blob_trigger(arg_name="blob", path="uploads/{name}", connection="AzureWebJobsStorage")
def analyze_image(blob: func.InputStream) -> None:
    analyze_blob(blob.name, blob.read())
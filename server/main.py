import json

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from ultralytics import YOLO

app = FastAPI()
model = YOLO("yolov8n.pt")


@app.post("/predict")
async def run_prediction(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid or unreadable image file")

    results = model.predict(img)

    parsed_results = json.loads(results[0].to_json())

    return {"detections": parsed_results}
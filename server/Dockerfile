FROM ultralytics/ultralytics:latest

WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir uv && uv pip install --system -r pyproject.toml
COPY *.py ./
COPY static ./static
COPY trackers ./trackers
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolo11n.pt')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
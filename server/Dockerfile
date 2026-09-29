FROM ultralytics/ultralytics:latest

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY main.py .
COPY static ./static

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
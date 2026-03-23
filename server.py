from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from ultralytics import YOLO

app = FastAPI()

model = YOLO("yolov8n.pt")


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    data = await file.read()

    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)

    names = results[0].names
    boxes = results[0].boxes

    detected = []

    if boxes is not None:
        for b in boxes:
            cls = int(b.cls[0])
            detected.append(names[cls])

    return {"detected": detected}
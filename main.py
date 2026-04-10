from fastapi import FastAPI, UploadFile, File, Query
from inference.predict import predict_image
from PIL import Image
import io
import time
import base64
import cv2
import numpy as np
import traceback

app = FastAPI()


@app.get("/")
def home():
    return {"message": "API running"}


# =========================
# EXPLAIN ENDPOINT
# =========================
@app.post("/explain")
async def explain(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        result = predict_image(image, return_heatmap=True)
        heatmap = result.get("heatmap")

        if heatmap is None:
            return {"error": "Heatmap is None"}

        heatmap = np.array(heatmap)

        print("DEBUG heatmap:", type(heatmap), heatmap.shape, heatmap.dtype)

        if heatmap.dtype != np.uint8:
            heatmap = (heatmap * 255).clip(0, 255).astype("uint8")

        success, buffer = cv2.imencode(".jpg", heatmap)

        if not success:
            return {"error": "Failed to encode heatmap"}

        heatmap_base64 = base64.b64encode(buffer).decode("utf-8")

        return {"heatmap": heatmap_base64}

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# =========================
# PREDICT ENDPOINT
# =========================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    explain: bool = Query(False)
):
    try:
        start = time.time()

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        result = predict_image(image, return_heatmap=explain)

        # 🔥 SAFETY CHECK
        if explain and result.get("heatmap") is None:
            return {"error": "GradCAM failed — heatmap is None"}

        latency = time.time() - start

        return {
            "prediction": result,
            "latency": latency
        }

    except Exception as e:
        print("\n🔥🔥 ERROR OCCURRED 🔥🔥")
        traceback.print_exc()
        return {"error": str(e)}
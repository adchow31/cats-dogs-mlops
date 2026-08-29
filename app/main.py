from fastapi import FastAPI, File, HTTPException, UploadFile

from app.schemas import HealthResponse, PredictionResponse
from src.inference_utils import load_model, predict

app = FastAPI(title="Cats vs Dogs Classifier", version="1.0.0")

MODEL = None


@app.on_event("startup")
def startup_event():
    global MODEL
    try:
        MODEL = load_model("models/baseline_cnn.pt")
    except FileNotFoundError:
        MODEL = None  # server still starts; /predict will report unavailable


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", model_loaded=MODEL is not None)


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    try:
        result = predict(MODEL, image_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not process image: {e}")

    return PredictionResponse(**result)
import logging
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from prometheus_fastapi_instrumentator import Instrumentator

from app.schemas import HealthResponse, PredictionResponse
from src.inference_utils import load_model, predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cats-dogs-api")

app = FastAPI(title="Cats vs Dogs Classifier", version="1.0.0")

# Prometheus metrics — auto-tracks request count, latency, status codes per endpoint
Instrumentator().instrument(app).expose(app)

MODEL = None


@app.on_event("startup")
def startup_event():
    global MODEL
    try:
        MODEL = load_model("models/baseline_cnn.pt")
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        MODEL = None
        logger.warning("Model file not found at startup")


@app.get("/")
def root():
    return {"message": "Cats vs Dogs Classifier API — visit /docs for interactive testing"}


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", model_loaded=MODEL is not None)


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    start = time.time()

    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    try:
        result = predict(MODEL, image_bytes)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=422, detail=f"Could not process image: {e}")

    latency_ms = round((time.time() - start) * 1000, 2)
    logger.info(
        f"prediction label={result['label']} confidence={result['confidence']} "
        f"filename={file.filename} content_type={file.content_type} latency_ms={latency_ms}"
    )

    return PredictionResponse(**result)
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
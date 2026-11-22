from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.model import predict_class

app = FastAPI(title="Iris Prediction API")


class IrisFeatures(BaseModel):
    sepal_length: float = Field(ge=0, le=10)
    sepal_width: float = Field(ge=0, le=10)
    petal_length: float = Field(ge=0, le=10)
    petal_width: float = Field(ge=0, le=10)


class Prediction(BaseModel):
    predicted_class: str


@app.get("/")
def root():
    return {
        "service": "Iris Prediction API",
        "health": "/health",
        "predict": "/predict",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(features: IrisFeatures) -> Prediction:
    label = predict_class(
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    )
    return Prediction(predicted_class=label)

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException,Response
from pydantic import BaseModel, Field


# ------------------------------------------------------------
# Application paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "delivery_delay_model.pkl"


# ------------------------------------------------------------
# Load the trained model
# ------------------------------------------------------------

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    raise RuntimeError(
        f"Unable to load model from {MODEL_PATH}: {exc}"
    ) from exc


# ------------------------------------------------------------
# FastAPI application
# ------------------------------------------------------------

app = FastAPI(
    title="Logistics Late-Delivery Prediction API",
    description=(
        "Predicts whether a logistics order is likely "
        "to be delivered late."
    ),
    version="1.0.0"
)


# ------------------------------------------------------------
# Input schema
# ------------------------------------------------------------

class OrderInput(BaseModel):
    price: float = Field(ge=0)
    freight_value: float = Field(ge=0)
    product_weight_g: float = Field(ge=0)
    product_volume_cm3: float = Field(ge=0)
    number_of_items: int = Field(ge=1)
    estimated_delivery_days: int = Field(ge=0)

    purchase_month: int = Field(ge=1, le=12)
    purchase_weekday: int = Field(ge=0, le=6)
    purchase_hour: int = Field(ge=0, le=23)

    same_state: int = Field(ge=0, le=1)
    customer_state: str
    seller_state: str


FEATURE_COLUMNS = [
    "price",
    "freight_value",
    "product_weight_g",
    "product_volume_cm3",
    "number_of_items",
    "estimated_delivery_days",
    "purchase_month",
    "purchase_weekday",
    "purchase_hour",
    "same_state",
    "customer_state",
    "seller_state"
]


# ------------------------------------------------------------
# API endpoints
# ------------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Logistics late-delivery API is running",
        "documentation": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.post("/predict")
def predict(order: OrderInput):
    try:
        order_dictionary = order.model_dump()

        input_data = pd.DataFrame(
            [order_dictionary],
            columns=FEATURE_COLUMNS
        )

        prediction = int(
            model.predict(input_data)[0]
        )

        probability = float(
            model.predict_proba(input_data)[0][1]
        )

        if probability >= 0.70:
            risk_level = "high"
        elif probability >= 0.40:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "late_delivery_prediction": prediction,
            "late_delivery_probability": round(
                probability,
                4
            ),
            "risk_percentage": round(
                probability * 100,
                2
            ),
            "risk_level": risk_level
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}"
        ) from exc


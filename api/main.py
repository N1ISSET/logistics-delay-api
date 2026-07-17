from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field


# ============================================================
# APPLICATION PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "delivery_delay_model.pkl"
)

TEMPLATE_DIRECTORY = BASE_DIR / "templates"
STATIC_DIRECTORY = BASE_DIR / "static"


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

try:
    model = joblib.load(MODEL_PATH)

except Exception as exc:
    raise RuntimeError(
        f"Unable to load model from {MODEL_PATH}: {exc}"
    ) from exc


# ============================================================
# CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Logistics Late-Delivery Prediction System",
    description=(
        "Predicts the risk that a logistics order "
        "will be delivered late."
    ),
    version="1.0.0"
)


# ============================================================
# CONFIGURE TEMPLATES AND STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIRECTORY),
    name="static"
)

templates = Jinja2Templates(
    directory=TEMPLATE_DIRECTORY
)


# ============================================================
# INPUT SCHEMA
# ============================================================

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


# ============================================================
# WEB INTERFACE
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def prediction_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "service": "logistics-delay-prediction-api"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict_delivery(order: OrderInput):
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
            recommended_action = (
                "Immediately monitor the shipment and "
                "notify logistics staff."
            )

        elif probability >= 0.40:
            risk_level = "medium"
            recommended_action = (
                "Increase shipment monitoring and "
                "check carrier progress."
            )

        else:
            risk_level = "low"
            recommended_action = (
                "Continue normal order processing."
            )

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
            "risk_level": risk_level,
            "recommended_action": recommended_action
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}"
        ) from exc
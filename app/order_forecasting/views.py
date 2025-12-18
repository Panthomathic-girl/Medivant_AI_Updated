# app/order_pattern_forecasting/customer_forecasting/views.py
from fastapi import APIRouter, HTTPException, File, UploadFile
from datetime import datetime
import shutil
from pathlib import Path
from config import ModelConfig

# Local imports
from .schema import CustomerForecastResponse
from .schema import PredictRequest
from .service import predict_raw
from .service import fine_tune_with_new_csv
from .service import train_pro_independent_model

router = APIRouter(prefix="/order", tags=["Customer-Level Forecasting"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = PROJECT_ROOT / "Order_Pattern_Forecasting_dataset.csv"


@router.post("/train")
async def train_from_uploaded_csv(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")

    temp_path = PROJECT_ROOT / f"temp_upload_{datetime.now():%Y%m%d_%H%M%S}.csv"

    try:
        # Save uploaded file temporarily in root
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Replace the master dataset in root (clean replace)
        if DATASET_PATH.exists():
            DATASET_PATH.unlink()

        temp_path.replace(DATASET_PATH)

        # Train using the dataset now in root
        model_path = train_pro_independent_model(data_path=DATASET_PATH)

        size_mb = Path(model_path).stat().st_size / (1024 * 1024)

        return {
            "status": "success",
            "new_model_path": str(model_path),
            "dataset_used": str(DATASET_PATH),
            "model_size_mb": round(size_mb, 2),
            "message": (
                f"FULL TRAINING SUCCESS!\n"
                f"Model: {model_path}\n"
                f"Trained on: {DATASET_PATH.name} (project root)\n"
                f"Size: {size_mb:.2f} MB\n"
                f"Check console for detailed stats"
            ),
            "trained_at": datetime.now().isoformat()
        }    

    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


@router.post("/predict", response_model=CustomerForecastResponse)
async def predict_customer_orders(request: PredictRequest):
    year, month = request.year, request.month
    if not (1 <= month <= 12):
        raise HTTPException(status_code=400, detail="Month must be 1–12")

    raw = predict_raw(year, month)

    customer_dict = {}
    for p in raw["predictions"]:
        cust = p["customer_id"]
        prod = p["product_id"]
        qty = p["predicted_quantity"]
        customer_dict.setdefault(cust, {})[prod] = qty

    customer_orders = [
        {"customer_id": c, "products": p} for c, p in customer_dict.items()
    ]

    return {
        "forecast_month": f"{year}-{month:02d}",
        "total_predicted_orders": int(raw["total_predicted_orders"]),
        "total_customers_expected_to_order": len(customer_orders),
        "customer_orders": customer_orders[:100],
        "generated_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": f"Customer forecast for {month:02d}/{year}"
    }


@router.post("/retrain")
async def retrain_model(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files allowed")

    temp_path = PROJECT_ROOT / f"temp_retrain_{datetime.now():%Y%m%d_%H%M%S}.csv"

    try:
        # Save uploaded file temporarily in root
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Fine-tune using the uploaded file and the master dataset in root
        stats = fine_tune_with_new_csv(
            uploaded_csv_path=temp_path,
            master_dataset_path=DATASET_PATH
        )

        return {
            "success": True,
            "message": stats["message"],
            "new_model_path": stats["new_model_path"],
            "model_version": stats["model_version"],
            "historical_records": stats["historical_records"],
            "uploaded_valid_rows": stats["uploaded_valid_rows"],
            "total_records_used": stats["total_records_used"],
            "duplicates_removed": stats["duplicates_removed"],
            "training_samples": stats["training_samples"],
            "positive_orders": stats["positive_orders"],
            "duration_seconds": stats["duration_seconds"],
            "retrained_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
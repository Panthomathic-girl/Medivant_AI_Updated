# app/predictive_lead_score/views.py
import shutil, os
import uuid
from typing import List
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from .models import LeadScoreResponse, BatchPredictionRequest, BatchPredictionResponse
from app.predictive_lead_score.agent import lead_score_agent
from app.predictive_lead_score.utils import Finetune
import joblib
from config import ModelConfig

router = APIRouter(prefix="/lead", tags=["Lead Scoring"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# --- LAZY MODEL LOADING ---
_pipeline = None
_model_path = Path(ModelConfig.LEAD_MODEL_FILE)

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not _model_path.exists():
            raise HTTPException(
                status_code=503,
                detail="Model not trained yet. Please use POST /lead/train with some labeled data first."
            )
        try:
            _pipeline = joblib.load(_model_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    return _pipeline

@router.post("/predict", response_model=List[BatchPredictionResponse])
async def predict_batch(request: BatchPredictionRequest):
    try:
        if not request.leads:
            raise HTTPException(status_code=400, detail="No leads provided")

        # Lazy load the model only when needed
        pipeline = get_pipeline()

        # Convert to DataFrame
        df_raw = pd.DataFrame([lead.dict() for lead in request.leads])

        # Keep lead_id for response
        lead_ids = df_raw["lead_id"].tolist()
        df_for_model = df_raw.drop(columns=["lead_id"], errors="ignore")

        # APPLY SAME PREPROCESSING AS TRAINING
        df_processed = Finetune.preprocess_for_prediction(df_for_model)

        # Ensure column order matches training (critical!)
        if hasattr(pipeline, "feature_names_in_"):
            expected_cols = list(pipeline.feature_names_in_)
            df_processed = df_processed.reindex(columns=expected_cols, fill_value=0)

        # Predict
        probabilities = pipeline.predict_proba(df_processed)[:, 1]

        results = [
            BatchPredictionResponse(
                lead_id=lead_id,
                prediction="Likely to Close" if prob >= 0.5 else "Unlikely to Close",
                score=round(prob * 100, 2)  # Return as 0-100 percentage
            )
            for lead_id, prob in zip(lead_ids, probabilities)
        ]

        return results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/score", response_model=LeadScoreResponse)
async def predict_lead_score(audio: UploadFile = File(...)):
    filename = audio.filename or "unknown"
    ext = Path(filename).suffix.lower()

    if ext not in ModelConfig.ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    if audio.content_type not in ModelConfig.ALLOWED_MIME:
        raise HTTPException(status_code=400, detail=f"Invalid MIME: {audio.content_type}")

    content = await audio.read()
    if len(content) > ModelConfig.MAX_SIZE:
        raise HTTPException(status_code=413, detail="File too large (>20MB)")

    file_id = str(uuid.uuid4())
    temp_path = UPLOAD_DIR / f"{file_id}{ext}"

    try:
        temp_path.write_bytes(content)
        result = await lead_score_agent(str(temp_path), audio.content_type)

        if result is None:
            raise HTTPException(status_code=500, detail="Failed to parse LLM response")

        score_float = float(result.get("score", 0))

        return LeadScoreResponse(
            score=score_float,
            explanation=result.get("explanation", "No explanation provided."),
            filename=filename
        )
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass


@router.post("/train")
async def retrain(request: BatchPredictionRequest):
    if not request.leads:
        raise HTTPException(status_code=400, detail="No leads provided for training")

    try:
        leads_data = [lead.dict() for lead in request.leads]
        result = await Finetune.train_model(leads_data)

        global _pipeline
        _pipeline = None  # Clear cache to load new model next time

        return result

    except ValueError as ve:
        # This will now give clear message about missing/invalid target
        raise HTTPException(status_code=400, detail=f"Training data invalid: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
    
    
@router.post("/retrain")
async def finetune_existing_model(request: BatchPredictionRequest):
    if not request.leads:
        raise HTTPException(status_code=400, detail="No leads provided for retraining")

    try:
        leads_data = [lead.dict() for lead in request.leads]

        # Check if any lead has a target label
        sample_lead = leads_data[0]
        has_target = any(key in sample_lead for key in ModelConfig.POSSIBLE_TARGETS)
        if not has_target:
            raise HTTPException(
                status_code=400,
                detail=f"Retraining requires labeled data. Include at least one target field: {ModelConfig.POSSIBLE_TARGETS}"
            )

        result = await Finetune.retrain_model(leads_data)

        # Invalidate cache so next prediction uses the updated model
        global _pipeline
        _pipeline = None

        return result

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Retraining failed: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")














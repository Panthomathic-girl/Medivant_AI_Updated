# app/predictive_lead_score/models.py
from typing import TypedDict, List, Optional
from pydantic import BaseModel, Field, create_model
from config import ModelConfig

# Static responses
class LeadScoreResponse(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Lead score as float between 0 and 1")
    explanation: str
    filename: str

class LeadAnalysisResult(TypedDict):
    transcription: str
    score: float
    explanation: str

# ────────────────────── DYNAMIC LeadInput MODEL ──────────────────────
# Build field definitions: all features required except target fields (optional)
fields = {
    ModelConfig.ID_FEATURE: (str, ...),  # lead_id is required
}

# Add all features as required
for name, typ in ModelConfig.all_features().items():
    fields[name] = (typ, ...)

# Add target columns as Optional
for target_name in ModelConfig.POSSIBLE_TARGETS:
    fields[target_name] = (Optional[int], None)

# Dynamically create the model
LeadInput = create_model(
    "LeadInput",
    __base__=BaseModel,
    **fields
)

# Batch models
class BatchPredictionRequest(BaseModel):
    leads: List[LeadInput] # type: ignore

class BatchPredictionResponse(BaseModel):
    lead_id: str
    prediction: str
    score: float  # 0–100 percentage




















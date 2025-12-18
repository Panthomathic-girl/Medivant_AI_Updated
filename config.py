# config.py
from pathlib import Path
from dotenv import load_dotenv
import os
from typing import Dict, Literal


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

class Settings:
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
    MAX_AUDIO_SIZE_MB: int = int(os.getenv("MAX_AUDIO_SIZE_MB", "20"))
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    

    def __post_init__(self):
        if not self.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY required in .env")

class ModelConfig:
    LEAD_MODEL_FILE = "app/predictive_lead_score/models/deal_closure_model.pkl"
    
# ────────────────────── CENTRAL FEATURE DEFINITION ──────────────────────
    CATEGORICAL_FEATURES: Dict[str, type] = {
        "vertical": str,
        "territory": str,
        "lead_source": str,
        "product_stage": str,
        "expected_frequency": str,
    }

    NUMERICAL_FEATURES: Dict[str, type] = {
        "target_price": float,
        "proposed_price": float,
        "price_discount_pct": float,
        "expected_order_volume": float,
        "hod_approval": int,
        "emails_sent": int,
        "emails_opened": int,
        "calls_made": int,
        "meetings_held": int,
        "avg_response_time_hours": float,
        "last_contact_age_days": int,
        "complaint_logged": int,
        "buying_trend_percent": float,
        "previous_orders": int,
        "inactive_flag": int,
        "overdue_payments": int,
        "license_expiry_days_left": int,
        "training_completed": int,
        "deal_age_days": int,
    }

    # ID field (always required, not a feature)
    ID_FEATURE = "lead_id"

    # Possible target columns (flexible)
    POSSIBLE_TARGETS = ["won", "deal_closed", "target", "deal_won", "is_closed"]

    # Combine all features for easy access
    @classmethod
    def all_features(cls) -> Dict[str, type]:
        return {**cls.CATEGORICAL_FEATURES, **cls.NUMERICAL_FEATURES}

    @classmethod
    def all_feature_names(cls) -> list[str]:
        return list(cls.all_features().keys())
    
    # FIXED: Added "audio/wave"
    ALLOWED_MIME = {
        "audio/mp3", "audio/mpeg",
        "audio/wav", "audio/x-wav", "audio/wave",
        "audio/flac", "audio/x-flac",
        "audio/aac",
        "audio/ogg",
        "audio/webm",
    }
    ALLOWED_EXT = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".webm"}

    MAX_SIZE = 20 * 1024 * 1024  # 20 MB
    
    ORDER_FORECAST_MODEL_FILE = Path("app/order_forecasting/models/order_forecast_model.pkl") 
    

TASKTYPE = Literal["call", "email", "meeting", "administrative", "follow up"]
PriorityType = Literal["low", "medium", "high", "urgent"]
StatusType = Literal["not started", "in progress", "completed", "waiting", "deferred"]

TASK_TYPE_MAPPING = {
    "internal meeting": "meeting",
    "internal task": "administrative",
    "demo": "meeting",
    "presentation": "meeting",
    "follow-up": "follow up",
    "followup": "follow up",
    "send email": "email",
    "make a call": "call",
    "schedule call": "call",
    "admin": "administrative",
    "administrative task": "administrative",
}

settings = Settings()
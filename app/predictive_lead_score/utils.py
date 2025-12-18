# app/predictive_lead_score/utils.py
from typing import Dict, Any, List, Optional, TypedDict
from app.predictive_lead_score.models import LeadAnalysisResult
import json
import pandas as pd
import joblib
import os
import shutil
import re
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import ModelConfig
from typing import Any, Optional
import logging
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# Optional: If you're using actual protobuf objects from google-generativeai
try:
    from google.generativeai.types import GenerateContentResponse
    PROTO_AVAILABLE = True
except ImportError:
    PROTO_AVAILABLE = False

TARGET_MODEL_PATH = ModelConfig.LEAD_MODEL_FILE

# ────────────────────── FINETUNE CLASS ──────────────────────

class Finetune:

    @staticmethod
    async def auto_preprocess(df: pd.DataFrame):
        df = df.copy()
        
        # Find which target column actually has data
        target = None
        for col in ModelConfig.POSSIBLE_TARGETS:
            if col in df.columns:
                # Check if it has any non-null values
                if df[col].notna().any():
                    target = col
                    break

        if target is None:
            raise ValueError(
                "No valid target column found. For training, at least one lead must include a target field "
                "(e.g., 'deal_closed': 0 or 1, or 'won': 1, etc.) with values 0 or 1."
            )

        # Validate that target values are 0 or 1 (after dropping NaN)
        valid_values = df[target].dropna().astype(int, errors='ignore')
        if not valid_values.isin([0, 1]).all():
            raise ValueError(
                f"Target column '{target}' must contain only 0 or 1 values. "
                f"Found invalid values: {sorted(df[target].dropna().unique())}"
            )

        # Now safely convert — fill missing targets? Or drop rows?
        # Better: drop rows with missing target (common practice)
        if df[target].isna().any():
            print(f"[Warning] Dropping {df[target].isna().sum()} rows with missing target '{target}'")
            df = df.dropna(subset=[target])

        y = df[target].astype(int)
        X = df.drop(columns=[target] + [t for t in ModelConfig.POSSIBLE_TARGETS if t != target])

        # Drop ID columns
        X = X.drop(columns=[c for c in X.columns if "id" in c.lower()], errors="ignore")

        # Encode categoricals
        for col in X.select_dtypes(include=['object', 'string']).columns:
            X[col] = X[col].fillna("missing")
            X[col] = pd.factorize(X[col])[0]

        # Fill numeric missing values
        X = X.fillna(X.median(numeric_only=True)).fillna(0)

        return X, y

    # ────────────────────── PREPROCESS FOR INFERENCE ──────────────────────
    
    @staticmethod
    def preprocess_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Drop ID
        df = df.drop(columns=[c for c in df.columns if "id" in c.lower()], errors="ignore")

        # Drop any target columns that might be present (shouldn't affect prediction)
        df = df.drop(columns=ModelConfig.POSSIBLE_TARGETS, errors="ignore")

        # Encode only known categorical columns
        for col in ModelConfig.CATEGORICAL_FEATURES.keys():
            if col in df.columns:
                df[col] = df[col].fillna("missing")
                df[col] = pd.factorize(df[col])[0]

        # Fill numerics
        df = df.fillna(df.median(numeric_only=True)).fillna(0)

        return df
    
    
    

    # ────────────────────── TRAIN & REPLACE ROOT MODEL ──────────────────────
    @classmethod
    async def train_model(cls, leads: List[dict]) -> dict:
        if not leads:
            raise ValueError("No leads provided for training")

        df = pd.DataFrame(leads)

        X, y = await cls.auto_preprocess(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
        )

        model = RandomForestClassifier(
            n_estimators=600,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds, average='weighted', zero_division=0), 4),
            "recall": round(recall_score(y_test, preds, average='weighted', zero_division=0), 4),
            "f1_score": round(f1_score(y_test, preds, average='weighted'), 4),
            "total_samples": len(df),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Overwrite the production model
        joblib.dump(model, TARGET_MODEL_PATH)

        return {
            "status": "Model trained & deployed",
            "model_path": TARGET_MODEL_PATH,
            "metrics": metrics,
        }
        
        
    # ────────────────────── RETRAIN (FINETUNE) EXISTING MODEL ──────────────────────
    @classmethod
    async def retrain_model(cls, new_leads: List[dict]) -> dict:
        if not new_leads:
            raise ValueError("No new leads provided for retraining")

        df_new = pd.DataFrame(new_leads)

        # Preprocess new data to get X_new, y_new
        X_new, y_new = await cls.auto_preprocess(df_new)

        # Load existing model if it exists
        model_path = Path(TARGET_MODEL_PATH)
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                print("[Info] Loaded existing model for retraining")
            except Exception as e:
                raise ValueError(f"Failed to load existing model: {str(e)}")
        else:
            print("[Info] No existing model found. Starting fresh training for retrain.")
            model = RandomForestClassifier(
                n_estimators=600,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )

        # Partial fit / incremental training
        # RandomForestClassifier supports warm_start for incremental learning
        if hasattr(model, "warm_start") and model.warm_start:
            # Increase estimators and continue training
            model.n_estimators += 200  # Add more trees
        else:
            # Enable warm_start and set higher estimators
            model.warm_start = True
            model.n_estimators = max(model.n_estimators, 600) + 200

        # Fit on new data only (this adds to existing trees if warm_start=True)
        model.fit(X_new, y_new)

        # Optional: Evaluate on new data
        preds_new = model.predict(X_new)
        metrics = {
            "accuracy_on_new_data": round(accuracy_score(y_new, preds_new), 4),
            "precision_on_new_data": round(precision_score(y_new, preds_new, average='weighted', zero_division=0), 4),
            "recall_on_new_data": round(recall_score(y_new, preds_new, average='weighted', zero_division=0), 4),
            "f1_on_new_data": round(f1_score(y_new, preds_new, average='weighted'), 4),
            "new_samples_used": len(df_new),
            "total_estimators_now": model.n_estimators,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save updated model
        joblib.dump(model, TARGET_MODEL_PATH)

        return {
            "status": "Model successfully retrained (finetuned) with new data",
            "model_path": TARGET_MODEL_PATH,
            "metrics": metrics,
        }
        
        
# ────────────────────── ADVANCED FINETUNE (FOR UNDERFITTING CASES) ──────────────────────                

class AdvancedFinetune:
    """
    Enhanced preprocessing pipeline designed to improve model performance
    when baseline accuracy is low (<=80%) — helps reduce underfitting.
    """

    _scaler = MinMaxScaler()  # Class-level scaler (fitted during training)

    @staticmethod
    async def auto_preprocess(df: pd.DataFrame, fit_scaler: bool = True):
        """
        Advanced preprocessing with scaling, outlier handling, and robust encoding.
        Use fit_scaler=True during training/retraining, False during prediction if needed.
        """
        df = df.copy()

        # Find target column
        target = None
        for col in ModelConfig.POSSIBLE_TARGETS:
            if col in df.columns and df[col].notna().any():
                target = col
                break

        if target is None:
            raise ValueError(
                "No valid target column found. Must include at least one of: "
                f"{ModelConfig.POSSIBLE_TARGETS} with values 0/1."
            )

        # Validate target
        valid_values = df[target].dropna().astype(int, errors='ignore')
        if not valid_values.isin([0, 1]).all():
            raise ValueError(
                f"Target '{target}' contains invalid values: {sorted(df[target].dropna().unique())}. Only 0/1 allowed."
            )

        # Drop rows with missing target
        if df[target].isna().any():
            print(f"[Advanced] Dropping {df[target].isna().sum()} rows with missing target")
            df = df.dropna(subset=[target])

        y = df[target].astype(int)
        X = df.drop(columns=[target] + [t for t in ModelConfig.POSSIBLE_TARGETS if t != target])

        # Drop ID columns
        X = X.drop(columns=[c for c in X.columns if "id" in c.lower()], errors="ignore")

        # ────────────── OUTLIER HANDLING (Clip extreme values) ──────────────
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if X[col].std() > 0:  # Avoid division by zero
                # Clip to 1st and 99th percentile to reduce outlier impact
                lower = X[col].quantile(0.01)
                upper = X[col].quantile(0.99)
                X[col] = X[col].clip(lower, upper)

        # ────────────── CATEGORICAL ENCODING (same as before) ──────────────
        for col in X.select_dtypes(include=['object', 'string']).columns:
            X[col] = X[col].fillna("missing")
            X[col] = pd.factorize(X[col])[0]

        # ────────────── ADVANCED: MinMax Scaling on ALL numerical features ──────────────
        if numeric_cols.any():
            if fit_scaler:
                # Fit and transform during training
                X[numeric_cols] = AdvancedFinetune._scaler.fit_transform(X[numeric_cols])
            else:
                # Only transform (for future prediction use)
                X[numeric_cols] = AdvancedFinetune._scaler.transform(X[numeric_cols])

        # Final fill for any remaining NaNs (should be rare now)
        X = X.fillna(0)

        return X, y

    @staticmethod
    def preprocess_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
        """
        Use the fitted scaler from training time for consistent inference.
        Assumes AdvancedFinetune._scaler has already been fitted.
        """
        df = df.copy()

        # Drop ID and targets
        df = df.drop(columns=[c for c in df.columns if "id" in c.lower()], errors="ignore")
        df = df.drop(columns=ModelConfig.POSSIBLE_TARGETS, errors="ignore")

        # Outlier clipping (same bounds as training)
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if col in df.columns and df[col].std() > 0:
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df[col] = df[col].clip(lower, upper)

        # Categorical encoding
        for col in df.select_dtypes(include=['object', 'string']).columns:
            # Use same factorize logic — new categories become -1 (handled safely)
            codes, _ = pd.factorize(df[col].fillna("missing"))
            df[col] = codes

        # Scale using pre-fitted scaler
        if numeric_cols.any():
            df[numeric_cols] = AdvancedFinetune._scaler.transform(df[numeric_cols])

        df = df.fillna(0)
        return df

        # ────────────────────── TRAIN & REPLACE ROOT MODEL ──────────────────────
    @classmethod
    async def train_model(cls, leads: List[dict]) -> dict:
        if not leads:
            raise ValueError("No leads provided for training")

        df = pd.DataFrame(leads)

        X, y = await cls.auto_preprocess(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
        )

        model = RandomForestClassifier(
            n_estimators=600,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds, average='weighted', zero_division=0), 4),
            "recall": round(recall_score(y_test, preds, average='weighted', zero_division=0), 4),
            "f1_score": round(f1_score(y_test, preds, average='weighted'), 4),
            "total_samples": len(df),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Overwrite the production model
        joblib.dump(model, TARGET_MODEL_PATH)

        return {
            "status": "Model trained & deployed",
            "model_path": TARGET_MODEL_PATH,
            "metrics": metrics,
        }
        
        
    # ────────────────────── RETRAIN (FINETUNE) EXISTING MODEL ──────────────────────
    @classmethod
    async def retrain_model(cls, new_leads: List[dict]) -> dict:
        if not new_leads:
            raise ValueError("No new leads provided for retraining")

        df_new = pd.DataFrame(new_leads)

        # Preprocess new data to get X_new, y_new
        X_new, y_new = await cls.auto_preprocess(df_new)

        # Load existing model if it exists
        model_path = Path(TARGET_MODEL_PATH)
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                print("[Info] Loaded existing model for retraining")
            except Exception as e:
                raise ValueError(f"Failed to load existing model: {str(e)}")
        else:
            print("[Info] No existing model found. Starting fresh training for retrain.")
            model = RandomForestClassifier(
                n_estimators=600,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )

        # Partial fit / incremental training
        # RandomForestClassifier supports warm_start for incremental learning
        if hasattr(model, "warm_start") and model.warm_start:
            # Increase estimators and continue training
            model.n_estimators += 200  # Add more trees
        else:
            # Enable warm_start and set higher estimators
            model.warm_start = True
            model.n_estimators = max(model.n_estimators, 600) + 200

        # Fit on new data only (this adds to existing trees if warm_start=True)
        model.fit(X_new, y_new)

        # Optional: Evaluate on new data
        preds_new = model.predict(X_new)
        metrics = {
            "accuracy_on_new_data": round(accuracy_score(y_new, preds_new), 4),
            "precision_on_new_data": round(precision_score(y_new, preds_new, average='weighted', zero_division=0), 4),
            "recall_on_new_data": round(recall_score(y_new, preds_new, average='weighted', zero_division=0), 4),
            "f1_on_new_data": round(f1_score(y_new, preds_new, average='weighted'), 4),
            "new_samples_used": len(df_new),
            "total_estimators_now": model.n_estimators,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save updated model
        joblib.dump(model, TARGET_MODEL_PATH)

        return {
            "status": "Model successfully retrained (finetuned) with new data",
            "model_path": TARGET_MODEL_PATH,
            "metrics": metrics,
        }

# ────────────────────── GEMINI RESPONSE PARSING (unchanged) ──────────────────────
def parse_gemini_generate_content_response(response: Any) -> Optional[LeadAnalysisResult]:
    text_content = ""

    if PROTO_AVAILABLE and isinstance(response, GenerateContentResponse):
        if response.candidates and response.candidates[0].content.parts:
            text_content = response.candidates[0].content.parts[0].text

    elif isinstance(response, dict):
        candidates = response.get("candidates", [])
        if candidates and "content" in candidates[0]:
            parts = candidates[0]["content"].get("parts", [])
            if parts and "text" in parts[0]:
                text_content = parts[0]["text"]

    elif isinstance(response, str):
        json_match = re.search(r"```json\s*({.*?})\s*```", response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                return {
                    "transcription": parsed.get("transcription", ""),
                    "score": parsed.get("score", 0),
                    "explanation": parsed.get("explanation", "")
                }
            except json.JSONDecodeError:
                pass

        text_match = re.search(r'"text":\s*"([^"]+)"', response.replace('\n', ' '))
        if text_match:
            text_content = text_match.group(1).replace("\\n", "\n")

    if text_content.strip():
        json_block_match = re.search(r"```json\s*({.*?})\s*```", text_content, re.DOTALL)
        if json_block_match:
            try:
                data = json.loads(json_block_match.group(1))
                return {
                    "transcription": data.get("transcription", ""),
                    "score": float(data.get("score", 0)),
                    "explanation": data.get("explanation", "")
                }
            except (json.JSONDecodeError, ValueError):
                return None

        try:
            data = json.loads(text_content.strip())
            return {
                "transcription": data.get("transcription", ""),
                "score": float(data.get("score", 0)),
                "explanation": data.get("explanation", "")
            }
        except json.JSONDecodeError:
            pass

    return None


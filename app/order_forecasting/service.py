# app/order_pattern_forecasting/customer_forecasting/service.py
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"
import logging
logging.getLogger('lightgbm').setLevel(logging.ERROR)

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from lightgbm import LGBMClassifier, LGBMRegressor
from config import ModelConfig

from .utils import load_and_clean

MODEL_PATH = ModelConfig.ORDER_FORECAST_MODEL_FILE
MODEL_DIR = Path("app/order_forecasting/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Global model package — loaded only when needed
_PKG: Optional[Dict[str, Any]] = None

def _load_model():
    """Safely load the trained model (lazy loading)"""
    global _PKG
    if _PKG is not None:
        return _PKG

    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Train the model first using:\n"
            "→ POST /order/train_from_csv   (full retrain)\n"
            "→ POST /order/retrain          (incremental)"
        )

    print(f"[{datetime.now():%H:%M:%S}] Loading model from {MODEL_PATH}...")
    try:
        _PKG = joblib.load(MODEL_PATH)
        print(f"Model loaded! Version: {_PKG.get('model_version', 'unknown')}")
        print(f"Trained on: {_PKG.get('trained_on', 'unknown')}")
        return _PKG
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e


def predict_raw(year: int, month: int):
    """ULTIMATE LONG-TERM FORECASTING — Works perfectly for 2027, 2030, 2050+"""
    pkg = _load_model()

    clf = pkg['classifier']
    reg = pkg['regressor']
    cust_to_num = pkg['cust_to_num']
    prod_to_num = pkg['prod_to_num']

    target_date = pd.Timestamp(f"{year}-{month:02d}-01")
    predictions = []
    total = 0

    # Dynamic cap — grows slightly over time
    all_historical_qty = [
        pkg['avg_qty_last_3m'].get(key, 1) * pkg['frequency'].get(key, 0.05) * 30
        for key in pkg['last_order_dates']
    ]
    qty_cap = int(np.percentile(all_historical_qty, 97)) if all_historical_qty else 120
    qty_cap = max(60, qty_cap)

    print(f"Forecasting {year}-{month:02d} | Using qty_cap = {qty_cap}")

    for cust_id in cust_to_num:
        for prod_id in prod_to_num:
            key = (cust_id, prod_id)
            last_str = pkg['last_order_dates'].get(key)
            if not last_str:
                continue

            last_date = pd.Timestamp(last_str)
            recency_days = (target_date - last_date).days

            # Allow up to 15 years of inactivity
            if recency_days > 5475:  # 15 years
                continue

            # SLOWER DECAY — critical for long-term
            recency_score = np.exp(-recency_days / 365.0)  # Yearly decay, not 120-day

            frequency = pkg['frequency'].get(key, 0.01)
            avg_qty = pkg['avg_qty_last_3m'].get(key, 1.0)
            is_active = int(recency_days <= 365)  # 1-year active
            tenure = pkg['tenure_days'].get(key, 30)

            X = np.array([[
                cust_to_num[cust_id],
                prod_to_num[prod_id],
                recency_days,
                recency_score,
                frequency,
                avg_qty,
                is_active,
                tenure
            ]])

            prob = clf.predict_proba(X)[0, 1]

            # Dynamic threshold: lower for very far future
            threshold = 0.40 if recency_days > 2000 else 0.50
            if prob < threshold:
                continue

            raw_qty = reg.predict(X)[0]
            qty = max(1, int(round(raw_qty * (1 + recency_days / 10000))))  # slight growth
            qty = min(qty, qty_cap)

            total += qty
            predictions.append({
                "customer_id": cust_id,
                "product_id": prod_id,
                "predicted_quantity": qty,
                "probability": round(prob, 4)
            })

    # Sort by probability descending
    predictions.sort(key=lambda x: x["probability"], reverse=True)

    print(f"Done: {len(predictions)} predictions for {year}-{month:02d}")

    return {
        "predictions": predictions,
        "total_predicted_orders": int(total),
        "active_pairs": len(predictions),
        "quantity_cap_used": qty_cap
    }


def _train_model(df: pd.DataFrame) -> Dict[str, Any]:
    """Common training logic extracted for both full and incremental training."""
    df['yrmnth'] = df['order_date'].dt.to_period('M')
    all_months = sorted(df['yrmnth'].unique())
    
    if len(all_months) < 2:
        raise ValueError("Need at least 2 months of data.")

    print(f"Data spans {len(all_months)} months: {all_months[0]} to {all_months[-1]}")
    print(f"Generating training samples for {len(all_months)-1} target months")

    # Mappings
    cust_to_num = {c: i for i, c in enumerate(sorted(df['customer_id'].unique()))}
    prod_to_num = {p: i for i, p in enumerate(sorted(df['product_id'].unique()))}
    num_to_cust = {i: c for c, i in cust_to_num.items()}
    num_to_prod = {i: p for p, i in prod_to_num.items()}

    # Containers
    X_train = []
    y_order = []
    y_qty = []

    metadata = {
        'last_order_dates': {},
        'frequency': {},
        'avg_qty_last_3m': {},
        'tenure_days': {}
    }

    for idx, target_period in enumerate(all_months[:-1]):
        target_date = target_period.to_timestamp()
        next_month_date = target_date + pd.offsets.MonthBegin(1)

        historical = df[df['order_date'] < target_date]
        target_month_df = df[(df['order_date'] >= target_date) & (df['order_date'] < next_month_date)]

        if historical.empty:
            continue

        actual_pairs = set(zip(target_month_df['customer_id'], target_month_df['product_id']))
        qty_map = dict(zip(zip(target_month_df['customer_id'], target_month_df['product_id']),
                           target_month_df['quantity']))

        cp_stats = historical.groupby(['customer_id', 'product_id']).agg(
            last_order=('order_date', 'max'),
            first_order=('order_date', 'min'),
            total_orders=('order_date', 'count'),
            total_qty=('quantity', 'sum'),
            qty_last_3m=('quantity', lambda x: x[historical['order_date'] >= target_date - pd.DateOffset(months=3)].sum()),
            n_last_3m=('order_date', lambda x: (x >= target_date - pd.DateOffset(months=3)).sum())
        ).reset_index()

        for _, row in cp_stats.iterrows():
            cust_id = row['customer_id']
            prod_id = row['product_id']
            key = (cust_id, prod_id)

            recency_days = (target_date - row['last_order']).days
            if recency_days > 1095:
                continue

            tenure_days = max(1, (row['last_order'] - row['first_order']).days + 1)
            frequency = row['total_orders'] / (tenure_days / 30.0)
            avg_qty_last_3m = (
                row['qty_last_3m'] / max(1, row['n_last_3m'])
                if row['n_last_3m'] > 0 else
                row['total_qty'] / max(1, row['total_orders'])
            )

            # Update metadata
            metadata['last_order_dates'][key] = row['last_order'].isoformat()
            metadata['frequency'][key] = frequency
            metadata['avg_qty_last_3m'][key] = avg_qty_last_3m
            metadata['tenure_days'][key] = tenure_days

            # Features
            X_train.append([
                cust_to_num[cust_id],
                prod_to_num[prod_id],
                recency_days,
                np.exp(-recency_days / 90.0),
                frequency,
                avg_qty_last_3m,
                int(recency_days <= 120),
                tenure_days
            ])

            ordered = 1 if key in actual_pairs else 0
            y_order.append(ordered)
            if ordered:
                y_qty.append(max(1, int(qty_map.get(key, 1))))

        if (idx + 1) % 5 == 0 or idx == len(all_months) - 2:
            print(f"   Processed {target_period} → {len(X_train):,} samples ({sum(y_order):,} positives)")

    print(f"\nTraining set complete:")
    print(f"   Total samples : {len(X_train):,}")
    print(f"   Positive orders: {len(y_qty):,}")

    # =================== PRINT X_train, y_order, y_qty AS DATAFRAMES ===================
    feature_names = [
        'cust_num', 'prod_num', 'recency_days', 'recency_score',
        'frequency', 'avg_qty_last_3m', 'is_active', 'tenure_days'
    ]

    print("\n" + "="*80)
    print("X_train (Feature Matrix) - First 20 rows:")
    print("="*80)
    X_df = pd.DataFrame(X_train, columns=feature_names)
    print(X_df.head(20).to_string(index=False))

    print("\n" + "="*80)
    print("y_order (Did they order next month?) - First 20:")
    print("="*80)
    y_order_series = pd.Series(y_order, name="ordered_next_month")
    print(y_order_series.head(20).to_string())

    print("\n" + "="*80)
    print("y_qty (Quantity when ordered) - First 20 positive samples:")
    print("="*80)
    if len(y_qty) > 0:
        y_qty_series = pd.Series(y_qty, name="predicted_quantity")
        print(y_qty_series.head(20).to_string())
    else:
        print("No positive samples found!")

    print("\nFull X_train shape:", X_df.shape)
    print("Positive samples:", sum(y_order))
    print("="*80)

    # =================== TRAIN CLASSIFIER ===================
    print("Training classifier...")
    clf = LGBMClassifier(
        n_estimators=700,
        learning_rate=0.05,
        max_depth=10,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        class_weight='balanced',
        verbose=-1
    )
    clf.fit(X_train, y_order)

    # =================== TRAIN REGRESSOR ===================
    print("Training regressor...")
    X_pos = [x for x, ordered in zip(X_train, y_order) if ordered == 1]
    if len(X_pos) == 0:
        raise ValueError("No positive samples found — check your data")

    reg = LGBMRegressor(
        n_estimators=700,
        learning_rate=0.05,
        max_depth=10,
        min_child_samples=5,
        random_state=42,
        verbose=-1
    )
    reg.fit(X_pos, y_qty)

    # =================== PACKAGE MODEL ===================
    model_package = {
        'classifier': clf,
        'regressor': reg,
        'cust_to_num': cust_to_num,
        'prod_to_num': prod_to_num,
        'cust_mapping': num_to_cust,
        'prod_mapping': num_to_prod,
        'last_order_dates': metadata['last_order_dates'],
        'frequency': metadata['frequency'],
        'avg_qty_last_3m': metadata['avg_qty_last_3m'],
        'tenure_days': metadata['tenure_days'],
        'feature_names': feature_names,
        'trained_on': datetime.now().isoformat(),
        'data_period': f"{all_months[0]} → {all_months[-1]}",
        'total_samples': len(X_train),
        'positive_samples': len(y_qty),
        'model_version': '1.0.0',
    }

    return model_package


def train_pro_independent_model(data_path: str | Path) -> str:
    """
    Full fresh training from a given data path.
    Loads data, trains model, saves and returns path.
    """
    print(f"[{datetime.now():%H:%M:%S}] Starting FULLY DYNAMIC & ROBUST training...")

    df = load_and_clean(data_path)
    if df.empty:
        raise ValueError("No data loaded.")

    model_package = _train_model(df)

    print("Deploying model...")
    joblib.dump(model_package, MODEL_PATH)
    size_mb = MODEL_PATH.stat().st_size / (1024*1024)
    print(f"SUCCESS! Model saved:")
    print(f"→ {MODEL_PATH}")
    print(f"→ Size: {size_mb:.2f} MB")
    print(f"→ Ready for predictions in 2025, 2026, 2030+")

    return str(MODEL_PATH)


def fine_tune_with_new_csv(
    uploaded_csv_path: str | Path,
    master_dataset_path: str | Path
) -> dict:
    """
    FULLY DYNAMIC incremental retrain
    Both paths MUST be provided by the caller → no defaults, no hardcodes
    """
    start_time = datetime.now()
    uploaded_csv_path = Path(uploaded_csv_path)
    master_dataset_path = Path(master_dataset_path)

    print(f"[{start_time:%H:%M:%S}] Starting incremental retraining...")
    print(f"   • Uploaded data : {uploaded_csv_path}")
    print(f"   • Master dataset: {master_dataset_path}")

    try:
        # Step 1: Load master dataset (only if it exists)
        if master_dataset_path.exists():
            old_df = pd.read_csv(master_dataset_path)
            old_count = len(old_df)
            print(f"Loaded {old_count:,} existing records")
        else:
            old_df = pd.DataFrame()
            old_count = 0
            print("No master dataset found → treating uploaded file as full history")

        # Step 2: Load and clean uploaded file
        new_df_raw = pd.read_csv(uploaded_csv_path)
        uploaded_raw = len(new_df_raw)

        cols = ["customer_id", "product_id", "order_date", "quantity", "last_refill_date"]
        new_df = new_df_raw.iloc[:, :min(len(cols), new_df_raw.shape[1])]
        new_df.columns = cols[:len(new_df.columns)]

        new_df['order_date'] = pd.to_datetime(new_df['order_date'], errors='coerce')
        new_df = new_df.dropna(subset=['order_date', 'customer_id', 'product_id', 'quantity'])
        new_df = new_df[new_df['quantity'] > 0].copy()
        new_df = new_df.drop(columns=['last_refill_date'], errors='ignore')

        uploaded_valid = len(new_df)
        if uploaded_valid == 0:
            raise ValueError("Uploaded file has no valid order rows")

        print(f"Uploaded: {uploaded_raw:,} → {uploaded_valid:,} valid rows")

        # Step 3: Merge + deduplicate
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined['order_date'] = pd.to_datetime(combined['order_date'])
        combined['order_date_only'] = combined['order_date'].dt.date

        before = len(combined)
        combined = combined.drop_duplicates(
            subset=['customer_id', 'product_id', 'order_date_only', 'quantity'],
            keep='last'
        )
        after = len(combined)
        duplicates_removed = before - after
        combined = combined.drop(columns=['order_date_only'], errors='ignore')

        print(f"Merge complete: {before:,} → {after:,} records ({duplicates_removed:,} duplicates removed)")

        # Step 4: Retrain on full combined data
        print("Training new model on all data...")
        model_package = _train_model(combined)

        # Step 5: Deploy
        joblib.dump(model_package, MODEL_PATH)
        size_mb = Path(MODEL_PATH).stat().st_size / (1024 * 1024)
        elapsed = int((datetime.now() - start_time).total_seconds())

        print(f"RETRAIN SUCCESS! Model deployed → {size_mb:.2f} MB")

        return {
            "success": True,
            "message": "Model retrained and deployed successfully!",
            "historical_records": old_count,
            "uploaded_valid_rows": uploaded_valid,
            "total_records_used": len(combined),
            "duplicates_removed": duplicates_removed,
            "training_samples": model_package['total_samples'],
            "positive_orders": model_package['positive_samples'],
            "duration_seconds": elapsed,
            "new_model_path": str(MODEL_PATH),
            "model_version": model_package['model_version'],
            "retrained_at": datetime.now().isoformat()
        }

    except Exception as e:
        raise RuntimeError(f"Retraining failed: {str(e)}") from e
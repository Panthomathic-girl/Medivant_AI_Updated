# app/order_pattern_forecasting/customer_forecasting/utils.py

import pandas as pd
from pathlib import Path
from datetime import datetime

def load_and_clean(data_path: str | Path) -> pd.DataFrame:
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    try:
        df = pd.read_csv(data_path)
    except Exception:
        df = pd.read_csv(data_path, header=None)

    # Standardize columns
    cols = ["customer_id", "product_id", "order_date", "quantity", "last_refill_date"]
    if df.shape[1] >= 4:
        df = df.iloc[:, :len(cols)]
        df.columns = cols[:df.shape[1]]
    else:
        df.columns = cols[:df.shape[1]]

    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    df = df.dropna(subset=['order_date', 'customer_id', 'product_id', 'quantity'])
    df = df[df['quantity'] > 0].copy()
    df = df.drop(columns=['last_refill_date'], errors='ignore')
    df = df.sort_values('order_date').reset_index(drop=True)

    print(f"Loaded {len(df):,} clean orders from {df['order_date'].min().date()} "
          f"to {df['order_date'].max().date()} → {data_path.name}")

    return df
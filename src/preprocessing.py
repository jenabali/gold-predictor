import pandas as pd
import numpy as np


def create_dataset(df: pd.DataFrame, window: int = 5):
    """Create sliding window dataset from OHLC data."""
    features = []
    labels = []
    for i in range(len(df) - window):
        window_df = df.iloc[i : i + window]
        next_close = df.iloc[i + window]["Close"]
        last_close = df.iloc[i + window - 1]["Close"]
        feature = window_df[["Open", "High", "Low", "Close"]].values.flatten()
        label = 1 if next_close > last_close else 0
        features.append(feature)
        labels.append(label)
    return np.array(features), np.array(labels)


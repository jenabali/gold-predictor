import yfinance as yf
import pandas as pd


def load_data(symbol: str = "XAUUSD=X", period: str = "10y", interval: str = "1h") -> pd.DataFrame:
    """Load historical data for the given symbol."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    df = df.reset_index()
    df = df.rename(columns={"index": "Datetime"})
    df = df.dropna()
    return df


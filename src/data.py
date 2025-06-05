import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

def load_data(symbol: str = "XAUUSD", bars: int = 5000) -> pd.DataFrame:
    """Load historical H1 data for XAUUSD from MetaTrader 5."""
    
    if not mt5.initialize():
        raise RuntimeError("❌ MetaTrader5 initialization failed. مطمئن شو MT5 باز و حساب متصل شده.")

    rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_H1, datetime.now(), bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError("❌ هیچ دیتایی از MetaTrader دریافت نشد. نماد XAUUSD فعال نیست؟")

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # تغییر نام ستون‌ها برای سازگاری با باقی برنامه
    df.rename(columns={
        "time": "Datetime",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close"
    }, inplace=True)

    return df

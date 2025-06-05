import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

def load_data(symbol: str = "XAUUSD", bars: int = 5000) -> pd.DataFrame:
    """Load historical H1 data for XAUUSD from MetaTrader 5."""
    
    # اتصال به MT5
    if not mt5.initialize():
        raise RuntimeError("❌ MetaTrader5 initialization failed. لطفاً مطمئن شو MT5 بازه و وارد حساب شدی.")
    
    # گرفتن ۵۰۰۰ کندل ۱ ساعته گذشته
    rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_H1, datetime.now(), bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError("❌ داده‌ای از MetaTrader دریافت نشد. نماد XAUUSD فعال نیست؟")

    # تبدیل به DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    return df

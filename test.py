import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import date, timedelta

DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "NVDA", "META", "NFLX", "AMD", "INTC",
    "ORCL", "IBM", "CSCO", "ADBE", "CRM",
    "PYPL", "UBER", "LYFT", "SHOP", "PLTR",
    "ZM", "SNAP", "SPOT", "NUE", "KO",
    "PEP", "MCD", "NKE", "WMT", "DIS",
    "V", "MA", "JPM", "GS", "BAC",
    "C", "WFC", "BA", "CAT", "GE",
    "MMM", "XOM", "CVX", "PFE", "JNJ",
    "MRK", "ABBV", "T", "F", "GM",
]

def _price_columns() -> list[str]:
    return ["Datetime", "Symbol", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]

def _download_symbol(symbol: str, start: date, end: date) -> pd.DataFrame:
    # 1. Đảm bảo logic ngày kết thúc (Exclusive end date)
    # if start >= end:
    #     end = start + timedelta(days=1)

    # 2. Download dữ liệu (Để yfinance tự lo khoản chống block, KHÔNG truyền session)
    frame = yf.download(
        symbol,
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=False,
        actions=True,
        progress=False,
        threads=False,
    )
    
    if frame.empty:
        return pd.DataFrame(columns=_price_columns())

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    frame = frame.reset_index()
    if "Date" in frame.columns:
        frame = frame.rename(columns={"Date": "Datetime"})
    frame["Datetime"] = pd.to_datetime(frame["Datetime"]).dt.tz_localize(None).dt.normalize()
    frame["Symbol"] = symbol
    
    for column in _price_columns():
        if column not in frame.columns:
            frame[column] = 0 if column in ["Dividends", "Stock Splits"] else np.nan
            
    return frame[_price_columns()]

if __name__ == "__main__":
    # Chọn 3 mã để test cho nhanh, bạn có thể đổi thành DEFAULT_SYMBOLS để test toàn bộ
    test_symbols = DEFAULT_SYMBOLS[:3] 
    
    today = date.today()
    
    print("="*50)
    print("BẮT ĐẦU TEST YFINANCE DOWNLOADER")
    print("="*50)
    
    # ==========================================
    # TEST 1: KỊCH BẢN INCREMENTAL (Lấy 1 ngày)
    # ==========================================
    print("\n[TEST 1] Chế độ Incremental (Lấy dữ liệu của ngày hôm qua/hôm nay):")
    # Giả lập Airflow truyền vào cùng 1 ngày, ví dụ: start = end = today
    start_incr = today
    end_incr = today
    print(f"Tham số đầu vào: start={start_incr}, end={end_incr}")
    
    for sym in test_symbols:
        df_incr = _download_symbol(sym, start_incr, end_incr)
        print(f" -> {sym}: Tải về {len(df_incr)} dòng.")
        if not df_incr.empty:
            # In thử 1 dòng để kiểm tra format
            print(df_incr[['Datetime', 'Symbol', 'Close', 'Volume']].head(1).to_string(index=False))
        else:
            print(f" -> {sym}: Không có dữ liệu (Thị trường đóng cửa hoặc lỗi).")

    # ==========================================
    # TEST 2: KỊCH BẢN BACKFILL (Lấy nhiều ngày)
    # ==========================================
    print("\n[TEST 2] Chế độ Backfill (Lấy dữ liệu 10 ngày qua):")
    start_backfill = date(2025, 3, 22)
    end_backfill = date(2026, 5, 27)
    print(f"Tham số đầu vào: start={start_backfill}, end={end_backfill}")
    
    for sym in test_symbols:
        df_back = _download_symbol(sym, start_backfill, end_backfill)
        print(f" -> {sym}: Tải về {len(df_back)} dòng.")
        if not df_back.empty:
            # In dòng đầu và dòng cuối để kiểm tra ngày tháng
            print("    [Dòng cũ nhất]")
            print(df_back[['Datetime', 'Symbol', 'Close', 'Volume']].head(1).to_string(index=False))
            print("    [Dòng mới nhất]")
            print(df_back[['Datetime', 'Symbol', 'Close', 'Volume']].tail(1).to_string(index=False))
        else:
            print(f" -> {sym}: Không có dữ liệu.")
            
    print("\n" + "="*50)
    print("HOÀN THÀNH TEST")
    print("="*50)
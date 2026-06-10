import json
import os
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytz
import requests
import yfinance as yf
from dotenv import load_dotenv
from kafka import KafkaProducer
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

# --- KAFKA CONFIG ---
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
MARKET_TOPIC = os.getenv("MARKET_TOPIC", "stock_market")
MARKET_INDICATOR_TOPIC = os.getenv("MARKET_INDICATOR_TOPIC", "stock_market_indicator")
NEWS_TOPIC = os.getenv("NEWS_TOPIC", "stock_news")
FINNHUB_API_KEY = os.getenv("API_KEY")

# --- MOCK CONFIG (Dành cho Giá & Chỉ số) ---
TARGET_DATE = "2026-06-09"  # Ngày giả lập dữ liệu giá
NEXT_DATE = "2026-06-10"    # Ngày tiếp theo
TICK_INTERVAL_SECONDS = 60   # Tốc độ phát 1 nến trong lúc Demo

SYMBOLS = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AGN", "AIG", "ALL", "AMD", "AMGN",
    "AMZN", "AXP", "BA", "BAC", "BIIB", "BK", "BLK", "BMY", "BRK.B", "C",
    "CAT", "CELG", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
    "CVX", "DD", "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F",
    "FDX", "FOX", "FOXA", "GD", "GE", "GILD", "GM", "GOOG", "GOOGL", "GS",
    "HAL", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KHC", "KMI", "KO",
    "LLY", "LMT", "LOW", "LYFT", "MA", "MCD", "MDLZ", "MDT", "MET", "META",
    "MMM", "MO", "MON", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NUE",
    "NVDA", "ORCL", "OXY", "PEP", "PFE", "PG", "PLTR", "PM", "PYPL",
    "QCOM", "RTN", "SBUX", "SHOP", "SLB", "SNAP", "SO", "SPG", "SPOT", "T",
    "TGT", "TSLA", "TWX", "TXN", "UBER", "UNH", "UNP", "UPS", "USB", "UTX",
    "V", "VZ", "WBA", "WFC", "WMT", "XOM", "ZM",
]

MARKET_INDICATORS = {
    "sp500": "^GSPC",
    "dow": "^DJI",
    "nasdaq": "^IXIC",
    "russell2000": "^RUT",
    "vix": "^VIX",
    "dxy": "DX-Y.NYB",
    "us10y": "^TNX",
    "us5y": "^FVX",
    "us3m": "^IRX",
    "oil_wti": "CL=F",
    "gold": "GC=F",
    "bitcoin": "BTC-USD",
    "sox": "^SOX",
}

producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BROKER],
    key_serializer=lambda key: key.encode("utf-8"),
    value_serializer=lambda value: json.dumps(value, default=str).encode("utf-8"),
)

market_buffer = defaultdict(list)
indicator_buffer = defaultdict(list)

# ---------------------------------------------------------
# PHẦN 1: STOCK NEWS (Chạy Real-time trên luồng riêng)
# ---------------------------------------------------------
def fetch_news(symbol: str, from_date: str, to_date: str) -> list[dict]:
    if not FINNHUB_API_KEY:
        logger.warning("Missing API_KEY in environment. Skip fetching news.")
        return []

    try:
        response = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={
                "symbol": symbol,
                "from": from_date,
                "to": to_date,
                "token": FINNHUB_API_KEY,
            },
            timeout=10,
        )
        response.raise_for_status()

        records = []
        for item in response.json():
            published_at = datetime.fromtimestamp(item["datetime"], tz=timezone.utc)
            records.append(
                {
                    **item,
                    "Symbol": symbol,
                    "Datetime": published_at.isoformat(),
                }
            )
        return records
    except Exception as e:
        logger.debug(f"Lỗi fetch news {symbol}: {e}")
        return []

def produce_news():
    """Lấy tin tức theo ngày thực tế hiện tại."""
    today = datetime.now(timezone.utc).date()
    from_date = os.getenv("NEWS_FROM_DATE", (today - timedelta(days=1)).isoformat())
    to_date = os.getenv("NEWS_TO_DATE", today.isoformat())

    for symbol in SYMBOLS:
        try:
            records = fetch_news(symbol, from_date, to_date)
            for record in records:
                producer.send(NEWS_TOPIC, key=record.get("Symbol"), value=record)
            
            if records:
                logger.info(f"📰 Sent {len(records)} real-time news records for {symbol}")
        except Exception as exc:
            logger.error(f"Failed news for {symbol}: {exc}")
        
        # Ngủ 1 giây để lách Rate Limit của API
        time.sleep(1)

def run_news_ingestion_thread():
    """Vòng lặp ngầm: Lấy tin tức định kỳ không làm nghẽn luồng chính."""
    interval_seconds_news = int(os.getenv("NEWS_INGESTION_INTERVAL_SECONDS", "300"))
    while True:
        logger.info(">>> Đang kích hoạt luồng tải tin tức Real-time...")
        produce_news()
        logger.info(f">>> Luồng tin tức hoàn tất. Chờ {interval_seconds_news} giây cho đợt tiếp theo.")
        time.sleep(interval_seconds_news)

# ---------------------------------------------------------
# PHẦN 2: MARKET DATA & INDICATOR (Mock Data)
# ---------------------------------------------------------
def load_symbol_data(symbol: str, is_indicator: bool = False):
    try:
        tk = yf.Ticker(symbol)
        df = tk.history(start=TARGET_DATE, end=NEXT_DATE, interval="1m")
        if df.empty:
            return 0
            
        df = df.reset_index()
        datetime_col = df.columns[0]
        count = 0
        
        for _, row in df.iterrows():
            dt_obj = pd.to_datetime(row[datetime_col])
            if dt_obj.tzinfo is None:
                dt_obj = dt_obj.tz_localize("America/New_York")
            else:
                dt_obj = dt_obj.tz_convert("America/New_York")
                
            time_key = dt_obj.strftime("%H:%M")
            dt_str = dt_obj.strftime("%Y-%m-%dT%H:%M:%S%z")
            
            # Lấy trước các giá trị để code gọn gàng hơn
            o = float(row.get("Open", 0.0)) if pd.notna(row.get("Open")) else 0.0
            h = float(row.get("High", 0.0)) if pd.notna(row.get("High")) else 0.0
            l = float(row.get("Low", 0.0)) if pd.notna(row.get("Low")) else 0.0
            c = float(row.get("Close", 0.0)) if pd.notna(row.get("Close")) else 0.0
            v = int(row.get("Volume", 0)) if pd.notna(row.get("Volume")) else 0
            d = float(row.get("Dividends", 0.0)) if pd.notna(row.get("Dividends")) else 0.0
            s = float(row.get("Stock Splits", 0.0)) if pd.notna(row.get("Stock Splits")) else 0.0

            if is_indicator:
                clean_record = {
                    "Open": o,
                    "High": h,
                    "Low": l,
                    "Close": c,
                    "Volume": v,
                    "Dividends": d,
                    "Stock Splits": s,
                    "Datetime": dt_str,
                    "Indicator": symbol
                }
                indicator_buffer[time_key].append(clean_record)
            else:
                clean_record = {
                    "Datetime": dt_str,
                    "Open": o,
                    "High": h,
                    "Low": l,
                    "Close": c,
                    "Volume": v,
                    "Dividends": d,
                    "Stock Splits": s,
                    "Symbol": symbol
                }
                market_buffer[time_key].append(clean_record)
            
            count += 1
        return count
    except Exception as e:
        logger.debug(f"Lỗi khi load data {symbol}: {e}")
        return 0

def preload_all_data():
    logger.info(f"Đang tải trước dữ liệu giả lập cho ngày {TARGET_DATE}...")
    total_market_records = 0
    total_indicator_records = 0

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_sym = {executor.submit(load_symbol_data, sym, False): sym for sym in SYMBOLS}
        for future in as_completed(future_to_sym):
            total_market_records += future.result()

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ind = {executor.submit(load_symbol_data, ind, True): ind for ind in MARKET_INDICATORS.values()}
        for future in as_completed(future_to_ind):
            total_indicator_records += future.result()

    logger.info(f"✅ Hoàn tất tải dữ liệu Market! Stocks: {total_market_records} dòng, Indicators: {total_indicator_records} dòng.")

def run_simulation():
    current_time = datetime.strptime("09:30", "%H:%M")
    end_time = datetime.strptime("16:00", "%H:%M")
    
    logger.info(f"🚀 BẮT ĐẦU CHẠY GIẢ LẬP REAL-TIME. Interval: {TICK_INTERVAL_SECONDS} giây.")
    
    while current_time <= end_time:
        time_key = current_time.strftime("%H:%M")
        
        market_records = market_buffer.get(time_key, [])
        for record in market_records:
            producer.send(MARKET_TOPIC, key=record["Symbol"], value=record)
            
        indicator_records = indicator_buffer.get(time_key, [])
        for record in indicator_records:
            producer.send(MARKET_INDICATOR_TOPIC, key=record["Indicator"], value=record)
            
        logger.info(f"[{time_key}] Đã gửi {len(market_records)} Stocks và {len(indicator_records)} Indicators.")
        
        time.sleep(TICK_INTERVAL_SECONDS)
        current_time += timedelta(minutes=1)

    logger.info("🏁 Đã phát xong toàn bộ dữ liệu phiên giao dịch.")

def cleanup():
    try:
        producer.flush()
        producer.close()
        logger.info("Kafka producer closed")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    try:
        # 1. Khởi chạy luồng News ngầm (Sẽ chạy song song và tự lặp lại sau mỗi 5 phút)
        news_thread = threading.Thread(target=run_news_ingestion_thread, daemon=True)
        news_thread.start()
        
        # 2. Load dữ liệu Market vào RAM (Luồng chính)
        preload_all_data()
        
        # 3. Bắt đầu phát dữ liệu Market theo từng phút (Luồng chính)
        run_simulation()
        
    except KeyboardInterrupt:
        logger.info("Simulation stopped by user")
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}")
    finally:
        cleanup()
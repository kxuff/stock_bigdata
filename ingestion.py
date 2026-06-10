import json
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytz
import requests
import yfinance as yf
from dotenv import load_dotenv
from kafka import KafkaProducer
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
MARKET_TOPIC = os.getenv("MARKET_TOPIC", "stock_market")
MARKET_INDICATOR_TOPIC = os.getenv("MARKET_INDICATOR_TOPIC", "stock_market_indicator")
NEWS_TOPIC = os.getenv("NEWS_TOPIC", "stock_news")
FINNHUB_API_KEY = os.getenv("API_KEY")

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


def fetch_market_data(symbol: str) -> dict | None:
    """
    Fetch market data for today only (during US market hours: 9:30 AM - 4:00 PM EDT)
    Falls back to latest available data if today's market is closed or has no data
    """
    try:
        stock = yf.Ticker(symbol)
        
        # Get today's date in US Eastern timezone
        eastern = pytz.timezone('America/New_York')
        today_eastern = datetime.now(eastern).date()
        
        # US Market hours: 9:30 AM - 4:00 PM EDT
        start_time = eastern.localize(datetime.combine(today_eastern, datetime.min.time().replace(hour=9, minute=30)))
        end_time = eastern.localize(datetime.combine(today_eastern, datetime.min.time().replace(hour=16, minute=0)))
        
        # Try to get today's data first
        data = stock.history(start=start_time, end=end_time, interval="1m")
        
        # If no data for today, fallback to latest available data (last 1 day)
        if data.empty:
            logger.debug(f"No intraday data for {symbol} today, fetching latest daily data")
            data = stock.history(period="1d", interval="1m")
        
        if data.empty:
            logger.debug(f"No market data available for {symbol}")
            return None

        latest = data.iloc[-2:-1]  # Get the most recent complete minute data (exclude the last row which may be incomplete)
        if latest.empty:
            return None

        latest = latest.reset_index()
        latest["Datetime"] = latest["Datetime"].astype(str)
        latest["Symbol"] = symbol
        return latest.to_dict(orient="records")[0]

        # data = data.reset_index()

        # Normalize datetime
        # data["Datetime"] = data["Datetime"].astype(str)

        # Add symbol column
        # data["Symbol"] = symbol

        # Convert dataframe -> list[dict]
        # records = data.to_dict(orient="records")

        # return records

    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {e}")
        return None

def fetch_market_indicator(indicator: str) -> dict | None:
    try:
        stock = yf.Ticker(indicator)

        eastern = pytz.timezone("America/New_York")
        today_eastern = datetime.now(eastern).date()

        start_time = eastern.localize(
            datetime.combine(
                today_eastern,
                datetime.min.time().replace(hour=9, minute=30)
            )
        )

        end_time = eastern.localize(
            datetime.combine(
                today_eastern,
                datetime.min.time().replace(hour=16, minute=0)
            )
        )

        data = stock.history(
            start=start_time,
            end=end_time,
            interval="1m"
        )

        if data.empty:
            print(
                f"No intraday data for {indicator}, "
                f"fetching latest available data"
            )

            data = stock.history(
                period="5d",
                interval="1m"
            )

        if data.empty:
            print(
                f"No market indicator data for {indicator}"
            )
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        # lấy cây nến hoàn chỉnh gần nhất
        latest = data.iloc[-2:-1]

        if latest.empty:
            return None

        latest = latest.reset_index()
        datetime_col = latest.columns[0]
        
        latest["Datetime"] = (
            pd.to_datetime(latest[datetime_col])
            .dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        )

        # latest.drop(columns=[datetime_col], inplace=True)
        latest["Indicator"] = indicator

        return latest.to_dict(orient="records")[0]

    except Exception as e:
        print(
            f"Error fetching market indicator "
            f"for {indicator}: {e}"
        )
        return None

def fetch_news(symbol: str, from_date: str, to_date: str) -> list[dict]:
    if not FINNHUB_API_KEY:
        raise RuntimeError("Missing API_KEY in environment or .env")

    response = requests.get(
        "https://finnhub.io/api/v1/company-news",
        params={
            "symbol": symbol,
            "from": from_date,
            "to": to_date,
            "token": FINNHUB_API_KEY,
        },
        timeout=30,
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


def produce_market_data() -> None:
    for symbol in SYMBOLS:
        try:
            record = fetch_market_data(symbol)

            if not record:
                logger.debug(f"No market data for {symbol}")
                continue
            producer.send(
                MARKET_TOPIC,
                key=record.get("Symbol"),
                value=record
            )

            logger.info(
                f"Sent market record for {symbol}"
            )

        except Exception as exc:
            logger.error(f"Failed market data for {symbol}: {exc}")

def produce_market_indicator() -> None:
    for indicator in MARKET_INDICATORS.values():
        try:
            record = fetch_market_indicator(indicator)

            if not record:
                logger.debug(f"No market data for {indicator}")
                continue
            producer.send(
                MARKET_INDICATOR_TOPIC,
                key=record.get("Indicator"),
                value=record
            )
            logger.info(
                f"Sent market indicator record for {indicator}"
            )
            
        except Exception as exc:
            logger.error(f"Failed market indicator data for {indicator}: {exc}")
                
def produce_news() -> None:
    today = datetime.now(timezone.utc).date()
    from_date = os.getenv("NEWS_FROM_DATE", (today - timedelta(days=1)).isoformat())
    to_date = os.getenv("NEWS_TO_DATE", today.isoformat())

    for symbol in SYMBOLS:
        try:
            records = fetch_news(symbol, from_date, to_date)
            for record in records:
                producer.send(NEWS_TOPIC,key = record.get("Symbol"), value=record)
            logger.info(f"Sent {len(records)} news records for {symbol}")
        except Exception as exc:
            logger.error(f"Failed news for {symbol}: {exc}")

def cleanup():
        """Cleanup resources"""
        try:
            producer.flush()
            producer.close()
            logger.info("Kafka producer closed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

def run_ingestion() -> None:
    interval_seconds_stock = int(os.getenv("INGESTION_INTERVAL_SECONDS", "60"))  
    interval_seconds_news = int(os.getenv("NEWS_INGESTION_INTERVAL_SECONDS", "300"))  
    stock_last_run = datetime.min.replace(tzinfo=timezone.utc)
    news_last_run = datetime.min.replace(tzinfo=timezone.utc)
    while True:
        now = datetime.now(timezone.utc)
        if (now - stock_last_run).total_seconds() >= interval_seconds_stock:
            produce_market_data()
            produce_market_indicator()
            stock_last_run = now
        if (now - news_last_run).total_seconds() >= interval_seconds_news:
            produce_news()
            news_last_run = now
        time.sleep(5)

if __name__ == "__main__":
    try:
        run_ingestion()
    except KeyboardInterrupt:
        logger.info("Ingestion stopped by user")
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}")
    finally:
        cleanup()

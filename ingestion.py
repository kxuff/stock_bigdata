import json
import os
import time
from datetime import datetime, timedelta, timezone

import requests
import yfinance as yf
from dotenv import load_dotenv
from kafka import KafkaProducer


load_dotenv()

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
MARKET_TOPIC = os.getenv("MARKET_TOPIC", "stock_market")
NEWS_TOPIC = os.getenv("NEWS_TOPIC", "stock_news")
FINNHUB_API_KEY = os.getenv("API_KEY")

SYMBOLS = [
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

producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BROKER],
    value_serializer=lambda value: json.dumps(value, default=str).encode("utf-8"),
)


def fetch_market_data(symbol: str) -> dict | None:
    stock = yf.Ticker(symbol)
    data = stock.history(period="1d", interval="1m")
    if data.empty:
        return None

    latest = data.iloc[:-1].tail(1)
    if latest.empty:
        latest = data.tail(1)

    latest = latest.reset_index()
    latest["Datetime"] = latest["Datetime"].astype(str)
    latest["Symbol"] = symbol
    return latest.to_dict(orient="records")[0]


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
                print(f"No market data for {symbol}")
                continue

            producer.send(MARKET_TOPIC, value=record)
            print(f"Sent market data: {symbol} | Close={record.get('Close')}")
        except Exception as exc:
            print(f"Failed market data for {symbol}: {exc}")


def produce_news() -> None:
    today = datetime.now(timezone.utc).date()
    from_date = os.getenv("NEWS_FROM_DATE", (today - timedelta(days=1)).isoformat())
    to_date = os.getenv("NEWS_TO_DATE", today.isoformat())

    for symbol in SYMBOLS:
        try:
            records = fetch_news(symbol, from_date, to_date)
            for record in records:
                producer.send(NEWS_TOPIC, value=record)
            print(f"Sent {len(records)} news records for {symbol}")
        except Exception as exc:
            print(f"Failed news for {symbol}: {exc}")


def run_once() -> None:
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{started_at}] Starting ingestion cycle")
    produce_market_data()
    produce_news()
    producer.flush()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished ingestion cycle")


if __name__ == "__main__":
    interval_seconds = int(os.getenv("INGESTION_INTERVAL_SECONDS", "300"))
    try:
        while True:
            run_once()
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("Stopping ingestion producer")
    finally:
        producer.close()

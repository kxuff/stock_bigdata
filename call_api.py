from kafka import KafkaProducer
import yfinance as yf
import pandas as pd
import json
import time
from datetime import datetime

# Cấu hình Kafka
KAFKA_BROKER = 'localhost:9092'  # Đổi lại nếu Docker Kafka của bạn map port khác
KAFKA_TOPIC = 'stock_market'  # Tên topic bạn muốn lưu dữ liệu

# Khởi tạo Kafka Producer
# value_serializer giúp tự động convert dictionary (dict) của Python thành JSON bytes
producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BROKER],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Danh sách mã cổ phiếu của bạn
symbols = [
    "AAPL","MSFT","GOOGL","AMZN","TSLA",
    "NVDA","META","NFLX","INTC","AMD",
    "KO","PEP","MCD","NKE","WMT",
    "DIS","V","MA","JPM","GS",
    "BABA","ORCL","IBM","CSCO","ADBE",
    "CRM","PYPL","UBER","LYFT","SHOP",
    "SQ","ZM","SNAP","TWTR","SPOT", # Lưu ý: TWTR có thể không trả về dữ liệu vì đã hủy niêm yết
    "SONY","TM","F","GM","BA",
    "GE","CAT","MMM","XOM","CVX",
    "PFE","JNJ","MRK","ABBV","T"
]

def fetch_and_produce():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Bắt đầu lấy và gửi dữ liệu...")
    
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            # Lấy dữ liệu 1 phút mới nhất (period="1d" là đủ, tail(1) để lấy dòng cuối cùng)
            data = stock.history(period="1d", interval="1m")
            data = data.iloc[:-1]

            latest = data.tail(1)
            
            if not latest.empty:
                # Đưa index (Datetime) thành cột dữ liệu
                latest.reset_index(inplace=True)
                
                # Format lại cột Datetime thành string để có thể serialize sang JSON
                latest['Datetime'] = latest['Datetime'].astype(str)
                latest['Symbol'] = symbol
                
                # Chuyển đổi DataFrame row thành Dictionary
                record = latest.to_dict(orient="records")[0]
                
                # Gửi message vào Kafka Topic
                producer.send(KAFKA_TOPIC, value=record)
                print(f"  -> Đã gửi: {symbol} | Giá: {record.get('Close')}")
            else:
                print(f"  -> Không có dữ liệu cho {symbol} lúc này (Có thể ngoài giờ giao dịch).")
                
        except Exception as e:
            print(f"  -> Lỗi khi lấy dữ liệu {symbol}: {e}")
    producer.flush()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Hoàn tất chu kỳ. Đang đợi chu kỳ tiếp theo...\n")

if __name__ == "__main__":
    print("Khởi động Kafka Producer cho Stock Data...")
    try:
        while True:
            fetch_and_produce()
            time.sleep(60)
    except KeyboardInterrupt:
        print("Đã dừng Producer do người dùng ngắt kết nối.")
    finally:
        producer.close()
import requests
import json
import time
from kafka import KafkaProducer
from dotenv import load_dotenv

# --- CẤU HÌNH ---
load_dotenv()
API_KEY = os.getenv("API_KEY")
SYMBOL = "AAPL"
KAFKA_BOOTSTRAP_SERVERS = ['localhost:9092']  # Thay đổi nếu Docker host khác
TOPIC_NAME = 'stock_news'

# Khởi tạo Kafka Producer
# value_serializer giúp tự động chuyển dict sang JSON bytes
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def fetch_and_send():
    url = "https://finnhub.io/api/v1/company-news"
    params = {
    "symbol": "AAPL",
    "from": "2026-05-05",
    "to": "2026-05-05",
    "token": API_KEY
}

    try:
        # 1. Lấy dữ liệu từ API
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # 2. Gửi dữ liệu vào Kafka
        producer.send(TOPIC_NAME, value=data)
        producer.flush() # Đảm bảo dữ liệu được đẩy đi ngay lập tức
        
        print(f"[{time.strftime('%H:%M:%S')}] Đã gửi dữ liệu {SYMBOL} thành công tới Kafka.")

    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    print(f"Bắt đầu Producer cho topic: {TOPIC_NAME}...")
    while True:
        fetch_and_send()
        
        # Chờ 5 phút (300 giây)
        time.sleep(300)
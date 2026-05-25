### 1. RSI (Relative Strength Index)

- **Ý nghĩa:** Đo lường xem giá đang tăng hoặc giảm mạnh như thế nào (thường dùng RSI14).
- **Công thức tính:** $$RSI=100-\frac{100}{1+RS}$$
  _Trong đó:_ $RS=\frac{Average~Gain}{Average~Loss}$

---

### 2. EMA (Exponential Moving Average)

- **Ý nghĩa:** Dùng để đo lường xu hướng (trend) của thị trường. Các đường EMA thường được sử dụng là EMA9, EMA20 và EMA50. Ví dụ: Nếu đường EMA20 nằm trên đường EMA50 (EMA20 > EMA50), điều này biểu thị thị trường đang trong xu hướng tăng (uptrend).
- **Công thức cốt lõi:**
  $$EMA_{t}=Price_{t}\cdot k+EMA_{t-1}(1-k)$$
  _Trong đó:_ $k=\frac{2}{n+1}$

---

### 3. Volume (Khối lượng giao dịch)

- **Ý nghĩa:** Được sử dụng để phát hiện (detect) dòng tiền lớn tham gia vào thị trường hoặc các hoạt động giao dịch bất thường (unusual activity).
- **Cách tính phổ biến (RVOL - Relative Volume):**
  $$RVOL=\frac{Current~Volume}{Average~Volume}$$

---

### 4. Volatility (Độ biến động thị trường) & ATR

- **Ý nghĩa:** Volatility dùng để đo lường mức độ biến động của thị trường. Trong đó, ATR (Average True Range) là chỉ báo đo lường độ biến động phổ biến nhất trong trading.
- **Cách tính/ Yếu tố cấu thành:** Để tính toán ATR, người ta sử dụng các dữ liệu về giá bao gồm:
  - Mức giá cao nhất (High).
  - Mức giá thấp nhất (Low).
  - Giá đóng cửa của phiên trước đó (Previous Close).

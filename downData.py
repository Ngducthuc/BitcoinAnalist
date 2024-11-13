import ccxt
import pandas as pd
from datetime import datetime, timedelta
# Khởi tạo kết nối tới sàn giao dịch, ví dụ Binance
exchange = ccxt.binance()
# Thiết lập các tham số
symbol = 'BTC/USDT'  # Cặp giao dịch Bitcoin
timeframe = '1h'  # Khung thời gian: 1 giờ
limit = 500  # Số lượng nến dữ liệu tối đa mỗi lần tải (theo giới hạn của sàn giao dịch)
# Tính toán thời gian bắt đầu từ 2 tháng trước
end_date = datetime.now()
start_date = end_date - timedelta(days=240)
# Chuyển đổi thời gian sang timestamp (ms)
since = int(start_date.timestamp() * 1000)
# Tải dữ liệu
ohlcv_data = []
while since < int(end_date.timestamp() * 1000):
    # Tải dữ liệu
    ohlcv_batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    if not ohlcv_batch:
        break  # Nếu không có dữ liệu mới thì thoát vòng lặp
    ohlcv_data.extend(ohlcv_batch)
    since = ohlcv_batch[-1][0] + 1  # Cập nhật timestamp bắt đầu cho lần tải tiếp theo
# Chuyển dữ liệu sang DataFrame
df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
# Định dạng lại cột timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.strftime('%m/%d/%Y %I:%M:%S %p')
# Lưu vào file CSV
df.to_csv('bitcoin_data.csv', index=False)
print("Dữ liệu đã được lưu vào file 'bitcoin_data.csv'.")
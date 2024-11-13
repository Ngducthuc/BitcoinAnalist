import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import os
import joblib
import ccxt
from datetime import datetime, timezone
from telegram import Bot
import asyncio
TELEGRAM_TOKEN = '7718760664:AAEfBsOfzR96YcfyQO9hvNOPMHEZIogu4CY'
CHAT_ID = '859389644'
# Bước 1: Tải dữ liệu quá khứ
df = pd.read_csv('bitcoin_data.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=False, errors='coerce')
timestamps = df['Timestamp'].values
data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
# Chuẩn hóa dữ liệu cho tất cả các cột
scaler = MinMaxScaler(feature_range=(0, 1))
if os.path.exists('scaler.save'):
    scaler = joblib.load('scaler.save')
else:
    scaler.fit(data)
    joblib.dump(scaler, 'scaler.save')
scaled_data = scaler.transform(data)
# Bước 3: Chuẩn bị dữ liệu đầu vào và đầu ra cho mô hình
sequence_length = 60
X = []
y = []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i, 3])
X, y = np.array(X), np.array(y)
# Bước 4: Khởi tạo hoặc tải mô hình
if os.path.exists('bitcoin_model.keras'):
    model = load_model('bitcoin_model.keras')
else:
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=64, epochs=100)
    model.save('bitcoin_model.keras')
# Bước 5: Tải dữ liệu mới từ Binance từ đầu ngày hôm nay đến hiện tại
exchange = ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '1h'
limit = 500
start_of_day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
since = int(start_of_day.timestamp() * 1000)
ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
new_df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
new_data = new_df[['Open', 'High', 'Low', 'Close', 'Volume']].values
# Chuẩn hóa và chuẩn bị dữ liệu mới để huấn luyện thêm
new_scaled_data = scaler.transform(new_data)
new_X, new_y = [], []
for i in range(sequence_length, len(new_scaled_data)):
    new_X.append(new_scaled_data[i-sequence_length:i])
    new_y.append(new_scaled_data[i, 3])
new_X, new_y = np.array(new_X), np.array(new_y)
# Huấn luyện tiếp mô hình với dữ liệu mới
if new_X.shape[0] > 0:
    model.fit(new_X, new_y, batch_size=64, epochs=10)
    model.save('bitcoin_model_updated.keras')
else:
    print("Không có đủ dữ liệu mới để huấn luyện thêm.")
# Bước 6: Dự đoán giá với dữ liệu cập nhật
test_data = np.concatenate((scaled_data[-sequence_length:], new_scaled_data))
X_test = np.reshape(test_data[-sequence_length:], (1, sequence_length, X.shape[2]))
predicted_price = model.predict(X_test)
predicted_price = scaler.inverse_transform(np.array([[0, 0, 0, predicted_price[0][0], 0]]))[:, 3]
print("Giá dự đoán:", predicted_price[0])
# Bước 7: Vẽ biểu đồ giá thực tế và giá dự đoán
timestamps_valid = np.append(timestamps[len(data) - sequence_length:], pd.to_datetime(timestamps[-1]) + pd.Timedelta(hours=1))
timestamps_valid = pd.to_datetime(timestamps_valid)
train = data[:len(data) - sequence_length]
valid = data[len(data) - sequence_length:]
valid_close_prices = valid[:, 3]
valid_close_prices = np.append(valid_close_prices, predicted_price)
plt.figure(figsize=(14, 5))
plt.plot(timestamps, data[:, 3], color='blue', label='Giá thực tế')
plt.plot(timestamps_valid, valid_close_prices, color='red', label='Giá dự đoán')
plt.xlabel('Time')
plt.ylabel('Giá Bitcoin')
plt.legend()
plt.savefig('bitcoin_prediction.png')
# Hàm gửi ảnh qua Telegram Bot
async def send_telegram_photo():
    bot = Bot(token=TELEGRAM_TOKEN)
    with open('bitcoin_prediction.png', 'rb') as photo:
        await bot.send_photo(chat_id=CHAT_ID, photo=photo, caption=f"Giá dự đoán Bitcoin 1h tiếp theo: {predicted_price[0]:,.2f} USD")
# Gọi hàm gửi ảnh qua Telegram trong async
asyncio.run(send_telegram_photo())
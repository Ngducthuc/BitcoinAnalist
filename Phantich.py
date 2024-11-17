import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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
CHAT_ID = '-4578601279'
# Bước 1: Tải dữ liệu quá khứ
df = pd.read_csv('bitcoin_data.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
# Chuẩn hóa dữ liệu cho LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
if os.path.exists('scaler.save'):
    scaler = joblib.load('scaler.save')
else:
    scaler.fit(data)
    joblib.dump(scaler, 'scaler.save')
scaled_data = scaler.transform(data)
# Chuẩn bị dữ liệu cho LSTM
sequence_length = 60
X_lstm, y_lstm = [], []
for i in range(sequence_length, len(scaled_data)):
    X_lstm.append(scaled_data[i-sequence_length:i])
    y_lstm.append(scaled_data[i, 3])
X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
# Chuẩn bị dữ liệu cho Random Forest
X_rf, y_rf = data[:-1], data[1:, 3]  # Dùng dữ liệu tĩnh
X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
# Huấn luyện Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_rf_train, y_rf_train)
# Khởi tạo hoặc tải mô hình LSTM
if os.path.exists('bitcoin_model.keras'):
    lstm_model = load_model('bitcoin_model.keras')
else:
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_lstm.shape[1], X_lstm.shape[2])))
    lstm_model.add(LSTM(units=50, return_sequences=False))
    lstm_model.add(Dense(units=25))
    lstm_model.add(Dense(units=1))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_lstm, y_lstm, batch_size=64, epochs=100)
    lstm_model.save('bitcoin_model.keras')
# Bước 2: Tải dữ liệu mới từ Binance
exchange = ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '1m'
limit = 1000
start_of_day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
since = int(start_of_day.timestamp() * 1000)
try:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    new_df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    new_df['Timestamp'] = pd.to_datetime(new_df['Timestamp'], unit='ms')
    new_data = new_df[['Open', 'High', 'Low', 'Close', 'Volume']].values
    file_path = 'bitcoin_data.csv'
    if os.path.exists(file_path):
        new_df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        new_df.to_csv(file_path, mode='w', header=True, index=False)
except ccxt.NetworkError as e:
    print("Lỗi mạng khi kết nối đến Binance API:", e)
except ccxt.ExchangeError as e:
    print("Lỗi khi truy vấn dữ liệu từ Binance:", e)
except Exception as e:
    print("Có lỗi không xác định:", e)
# Chuẩn hóa dữ liệu mới
new_scaled_data = scaler.transform(new_data)
new_X, new_y = [], []
for i in range(sequence_length, len(new_scaled_data)):
    new_X.append(new_scaled_data[i-sequence_length:i])
    new_y.append(new_scaled_data[i, 3])
new_X, new_y = np.array(new_X), np.array(new_y)
# Huấn luyện tiếp mô hình với dữ liệu mới
if new_X.shape[0] > 0:
    lstm_model.fit(new_X, new_y, batch_size=64, epochs=10)
    lstm_model.save('bitcoin_model_updated.keras')
else:
    print("Không có đủ dữ liệu mới để huấn luyện thêm.")
X_test_lstm = np.reshape(new_scaled_data[-sequence_length:], (1, sequence_length, X_lstm.shape[2]))
lstm_prediction = lstm_model.predict(X_test_lstm)
lstm_prediction = scaler.inverse_transform(np.array([[0, 0, 0, lstm_prediction[0][0], 0]]))[:, 3]
rf_prediction = rf_model.predict([new_data[-1]])
final_prediction = (lstm_prediction[0] + rf_prediction[0]) / 2
plt.figure(figsize=(14, 5))
timestamps = df['Timestamp'].values
plt.plot(timestamps, data[:, 3], color='blue', label='Giá thực tế')
plt.scatter([timestamps[-1]], [final_prediction], color='red', label='Giá dự đoán cuối', zorder=5)
plt.xlabel('Time')
plt.ylabel('Giá Bitcoin')
plt.legend()
plt.title('Dự đoán giá Bitcoin')
plt.grid(True)
# Lưu hình ảnh
plt.savefig('bitcoin_prediction_combined.png')
async def send_telegram_photo():
    bot = Bot(token=TELEGRAM_TOKEN)
    with open('bitcoin_prediction_combined.png', 'rb') as photo:
        await bot.send_photo(chat_id=CHAT_ID, photo=photo, caption=f"Giá dự đoán Bitcoin: {final_prediction:,.2f} USD")
asyncio.run(send_telegram_photo())

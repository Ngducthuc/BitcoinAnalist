import subprocess
import asyncio
from telegram import Bot
# Chạy file phân tích tin tức
def run_news_analysis():
    subprocess.run(["python", "Tintuc.py"])
# Chạy file dự đoán giá Bitcoin và gửi dữ liệu qua Telegram
def run_price_prediction():
    subprocess.run(["python", "Phantich.py"])
# Thực hiện chuỗi các nhiệm vụ
async def main():
    # Chạy phân tích tin tức
    run_news_analysis()
    # Chạy dự đoán giá Bitcoin
    run_price_prediction()
# Chạy tất cả các nhiệm vụ
asyncio.run(main())

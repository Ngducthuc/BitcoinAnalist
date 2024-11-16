import subprocess
import asyncio
from telegram import Bot
def run_news_analysis():
    subprocess.run(["python", "Tintuc.py"])
def run_price_prediction():
    subprocess.run(["python", "Phantich.py"])
async def main():
    run_news_analysis()
    run_price_prediction()
asyncio.run(main())

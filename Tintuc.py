import requests
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from telegram import Bot
import asyncio
from deep_translator import GoogleTranslator
nltk.download('punkt')
nltk.download('stopwords')
TELEGRAM_TOKEN = '7718760664:AAEfBsOfzR96YcfyQO9hvNOPMHEZIogu4CY'
CHAT_ID = '859389644'
# Hàm lấy tin tức từ News API
def fetch_bitcoin_news(api_key, from_date='2024-11-12', to_date='2024-11-13'):
    url = f"https://newsapi.org/v2/everything?q=Bitcoin&from={from_date}&to={to_date}&sortBy=publishedAt&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        print("Lỗi khi kết nối với API")
        return []
    news_data = response.json()
    if news_data.get("status") != "ok" or not news_data.get("articles"):
        print("Lỗi lấy dữ liệu hoặc không có tin tức mới")
        return []
    articles = news_data["articles"]
    news_list = []
    for article in articles:
        news_list.append({
            "title": article.get("title", ""),
            "description": article.get("description", ""),
            "content": article.get("content", ""),
            "published_at": article.get("publishedAt", ""),
            "url": article.get("url", "")
        })
    return news_list
def preprocess_text(text):
    if not text:
        return ""
    text = re.sub(r'[^\w\s]', '', text.lower())
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)
# Dữ liệu mẫu để huấn luyện
data = {
    "text": ["Bitcoin price hits new high", "Bitcoin faces regulatory challenges", 
             "Bitcoin adoption is increasing", "Bitcoin market crashes due to new policies",
             "Bitcoin is booming in the financial world", "Bitcoin is facing challenges from the government",
             "Bitcoin price predictions are very optimistic", "Bitcoin crashes in value again"],
    "sentiment": [1, 0, 1, 0, 1, 0, 1, 0]  # 1 - Tốt, 0 - Xấu
}
df = pd.DataFrame(data)
df["text"] = df["text"].apply(preprocess_text)
# Chia dữ liệu và vector hóa
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["sentiment"], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)
# Huấn luyện mô hình
model = LogisticRegression()
model.fit(X_train_vectors, y_train)
# Hàm phân loại tin tức
def classify_news(news_list, vectorizer, model):
    if not news_list:
        print("Không có tin tức để phân loại.")
        return []
    classified_news = []
    for news in news_list:
        # Kiểm tra nếu title và description là None
        title = news.get("title", "")
        description = news.get("description", "")
        
        # Nối tiêu đề và tóm tắt với điều kiện không phải None
        full_text = (title if title else "") + " " + (description if description else "")
        
        if not full_text.strip():  # Kiểm tra nếu văn bản sau khi nối không rỗng
            continue
        
        # Tiền xử lý và phân loại tin tức
        processed_text = preprocess_text(full_text)
        text_vector = vectorizer.transform([processed_text])
        sentiment = model.predict(text_vector)[0]
        
        classified_news.append({
            "title": title,
            "description": description,
            "published_at": news.get("publishedAt", ""),
            "sentiment": "Positive" if sentiment == 1 else "Negative",
            "url": news.get("url", "")
        })
    return classified_news
# Hàm dịch tự động phát hiện ngôn ngữ và dịch sang tiếng Việt
def translate_to_vietnamese(text):
    try:
        # Dịch tự động phát hiện ngôn ngữ và dịch sang tiếng Việt
        translated_text = GoogleTranslator(source='auto', target='vi').translate(text)
        return translated_text
    except Exception as e:
        print(f"Lỗi khi dịch: {e}")
        return text  # Trả về nguyên văn nếu có lỗi
def analyze_market_trend_with_examples(classified_news):
    if not classified_news:
        print("Không có dữ liệu để phân tích xu hướng.")
        return "Không thể đưa ra xu hướng thị trường do thiếu dữ liệu."
    positive_news = [news for news in classified_news if news["sentiment"] == "Positive"]
    negative_news = [news for news in classified_news if news["sentiment"] == "Negative"]
    total_count = len(classified_news)
    positive_ratio = len(positive_news) / total_count
    negative_ratio = len(negative_news) / total_count
    if positive_ratio > 0.6:
        trend_analysis = "Thị trường Bitcoin có dấu hiệu tích cực với nhiều tin tức tốt."
        example_news = positive_news[:3]
    elif negative_ratio > 0.6:
        trend_analysis = "Thị trường Bitcoin đang gặp khó khăn với nhiều tin tức xấu."
        example_news = negative_news[:3]
    else:
        trend_analysis = "Thị trường Bitcoin có dấu hiệu biến động không rõ ràng với sự cân bằng giữa tin tức tốt và xấu."
        example_news = classified_news[:3]
    message = f"{trend_analysis}\n\nMột số tin tức tiêu biểu:\n"
    # Sửa lại phần in message
    for news in example_news:
        message += f"- Tiêu đề: {translate_to_vietnamese(news['title'])}\n"
        message += f"  Tóm tắt: {translate_to_vietnamese(news['description'])}\n"
        message += f"  Ngày xuất bản: {news['published_at']}\n"
        message += f"  Link: {news['url']}\n"
        message += f"  Trạng thái: {translate_to_vietnamese(news['sentiment'])}\n\n"
    print(message)
    return message
async def send_telegram_message(message):
    bot = Bot(token=TELEGRAM_TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text=message)
api_key = 'e5fb26ebebe0431482c255643b2a5ae6'
news_list = fetch_bitcoin_news(api_key)
classified_news = classify_news(news_list, vectorizer, model)
trend_message = analyze_market_trend_with_examples(classified_news)
asyncio.run(send_telegram_message(trend_message))
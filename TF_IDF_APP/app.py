from flask import Flask, render_template, request
import joblib
import re
from pathlib import Path

app = Flask(__name__)


def preprocess_for_tfidf(text):
    if isinstance(text, str):
        text = text.lower()
    return text

# 1. 載入模型
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "disaster_model.pkl"
vectorizer_path = BASE_DIR / "tfidf_vectorizer.pkl"
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# 2. 定義清理函數
def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r'http\S+|www\.\S+', '<URL>', text)
    text = re.sub(r'@\w+', '<USER>', text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip(' ')
    return text.lower()

# 3. 設定路由
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = ""
    result_color = ""
    input_text = ""

    if request.method == 'POST':
        input_text = request.form['tweet_text']
        
        if input_text:
            # 資料清理與預測
            cleaned = clean_text(input_text)
            vec_text = vectorizer.transform([cleaned])
            pred = model.predict(vec_text)[0]
            proba = model.predict_proba(vec_text)[0]

            # 設定顯示結果
            if pred == 1:
                conf = round(proba[1] * 100, 2)
                prediction_text = f" 這是災難推文 (信心指數: {conf}%)"
                result_color = "#d9534f"
            else:
                conf = round(proba[0] * 100, 2)
                prediction_text = f" 這是平安推文 (信心指數: {conf}%)"
                result_color = "#5cb85c" 

    return render_template('index.html', 
                           prediction=prediction_text, 
                           color=result_color, 
                           original_input=input_text)

if __name__ == '__main__':
    app.run(debug=True)
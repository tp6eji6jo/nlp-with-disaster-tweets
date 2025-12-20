# Natural Language Processing with Disaster Tweets

This project implements a **web-based inference platform** for the Kaggle task **Natural Language Processing with Disaster Tweets**.  
Users can input their own tweets and let the trained NLP model predict whether the tweet is related to a real disaster or not.

---

## 📌 Project Overview

- **Task**: Binary text classification  
  - `1` → Disaster-related tweet  
  - `0` → Non-disaster tweet
- **Model**: Trained on the *Disaster Tweets* dataset
- **Application**: Interactive web platform for real-time inference

---

## 🚀 How to Run the Platform

### 1️⃣ Start the application

Run the following command in the project directory:

```bash
python app.py
```

or (if using `uv`):

```bash
uv run app.py
```

---

### 2️⃣ Open the web interface

After execution, the terminal will display a message similar to:

```text
Running on http://127.0.0.1:5000
```

Copy this URL and paste it into your browser.

---

### 3️⃣ Try your own tweet

- Enter any tweet text into the input box  
- Submit the text  
- The platform will output the model’s prediction result

---

## 🧠 Platform Functionality

- Accepts **custom user input tweets**
- Applies the **same preprocessing pipeline** used during training
- Performs **real-time model inference**
- Displays prediction results directly on the web interface

---

## 📂 Project Structure

```text
.
├── app.py                  # Flask application entry point
├── disaster_model.pkl      # Trained NLP model
├── templates/
│   └── index.html          # Web interface
└── README.md
```

---

## 🔧 Requirements

- Python 3.x
- Flask
- scikit-learn
- joblib

---

## 🎯 Purpose

This project demonstrates how an NLP classification model can be **deployed as an interactive platform**, allowing users to experiment with real-world text inputs beyond static evaluation datasets.
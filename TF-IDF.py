# %%
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re

# %%
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
subm = pd.read_csv('data/sample_submission.csv')

# %%
def clean_text(text:  str) -> str:
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
    return text

train["clean_text"] = train["text"].apply(clean_text)
test["clean_text"] = test["text"].apply(clean_text)

y_train = train["target"]

# %%
def preprocess_for_tfidf(text):
    text = text.lower()
    return text

vec = TfidfVectorizer(
    preprocessor=preprocess_for_tfidf,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    strip_accents='unicode'
)

X_train = vec.fit_transform(train["clean_text"])
X_test  = vec.transform(test["clean_text"])

# %%
X_train, X_test

# %%
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# %%
y_test = clf.predict(X_test)
subm["target"] = y_test
subm.to_csv("submission_TF-IDF.csv", index=False)
# %%

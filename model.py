from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
import joblib

data = pd.read_csv(
    './data/spam.csv', encoding='latin-1')
data = data.rename(columns={'v1': 'target', 'v2': 'text'})
data = data[['text', 'target']]

ps = PorterStemmer()

corpus = []

for i in range(0, len(data)):
    msg = re.sub('[^a-zA-Z]', ' ', data['text'][i])
    msg = msg.lower()
    msg = msg.split()
    msg = [ps.stem(word)
           for word in msg if word not in stopwords.words('english')]
    msg = ' '.join(msg)
    corpus.append(msg)

cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(data['target'])['spam'].values
y

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, shuffle=True)


model = MultinomialNB().fit(X_train, y_train)
y_pred = model.predict(X_val)

print(classification_report(y_pred, y_val))

confusion_matrix(y_pred, y_val)

joblib.dump(model, 'model.pkl')
joblib.dump(cv, 'vectorizer.pkl')
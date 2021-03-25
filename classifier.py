import streamlit as st
import joblib
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

def classifier(msg):
    ps = PorterStemmer()
    msg = re.sub('[^a-zA-Z]', ' ', msg)
    msg = msg.lower()
    msg = msg.split()
    msg = [ps.stem(word)
           for word in msg if word not in stopwords.words('english')]
    msg = ' '.join(msg)
    print(msg)
    cv = joblib.load('vectorizer.pkl')
    dt = cv.transform([msg]).toarray()
    
    model = joblib.load('model.pkl')
    pred = model.predict(dt)
    spam_proba = model.predict_proba(dt)
    print(spam_proba)
    
    return [int(pred), "{:.2f}".format(float(spam_proba[:, 1]))]

def main():
    st.title('Spam Classifier')
    st.subheader('Using MultinomialNB Model trained on UCI SMS Spam Dataset')

    input = st.text_area('Enter text to classify', '')
    if input:
        x = classifier(input)
        if  x[0] == 0:
            st.write('NOT SPAM', x[1])
        else:
            st.write('SPAM', x[1])

if __name__ == '__main__':
    main()

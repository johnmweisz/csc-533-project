import os
import speech_recognition as sr
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import nltk
import pandas as pd
import re


def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def train_model():
    nltk.download('stopwords')
    df = pd.read_csv('data/labeled_data.csv')

    df['cleaned_tweet'] = df['tweet'].apply(preprocess_text)

    vectorizer = TfidfVectorizer(max_features=5000, stop_words=stopwords.words('english'))
    X = vectorizer.fit_transform(df['cleaned_tweet']).toarray()
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return vectorizer, model

def classify_text(vectorizer, model, text):
    cleaned_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([cleaned_text]).toarray()
    prediction = model.predict(text_vectorized)
    class_mapping = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
    return class_mapping[prediction[0]]

def audio_to_text(file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
            return recognizer.recognize_google(audio)

    except Exception as e:
        print(f"Error processing the audio file: {e}")
        return None

def main():
    vectorizer, model = train_model()

    folder_path = "data/audio"
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file: {file_path}")
            text = audio_to_text(file_path)
            if text is not None:
                result = classify_text(vectorizer, model, text)
                print(f"{filename} classified as: {result}")
            else:
                print(f"Could not process the audio file: {filename}")

if __name__ == "__main__":
    main()

import os
import logging
import speech_recognition as sr
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords, wordnet, words
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import joblib
import random
import csv

# Setup logging
logging.basicConfig(level=logging.INFO)

# Ensure NLTK data is downloaded
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('words', quiet=True)

def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

def train_model():
    df = pd.read_csv('data/labeled_data.csv')
    df['cleaned_tweet'] = df['tweet'].apply(preprocess_text)
    X = df['cleaned_tweet']
    y = df['class']
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression())
    ])
    param_grid = [
        {
            'tfidf__max_features': [3000, 5000, 10000],
            'tfidf__ngram_range': [(1,1), (1,2), (1,3)],
            'clf': [LogisticRegression(max_iter=1000)],
            'clf__C': [0.1, 1, 10, 100],
            'clf__class_weight': ['balanced']
        },
        {
            'tfidf__max_features': [3000, 5000, 10000],
            'tfidf__ngram_range': [(1,1), (1,2), (1,3)],
            'clf': [SVC()],
            'clf__C': [0.1, 1, 10, 100],
            'clf__kernel': ['linear', 'rbf'],
            'clf__class_weight': ['balanced']
        },
        {
            'tfidf__max_features': [3000, 5000, 10000],
            'tfidf__ngram_range': [(1,1), (1,2), (1,3)],
            'clf': [RandomForestClassifier()],
            'clf__n_estimators': [100, 200],
            'clf__class_weight': ['balanced']
        }
    ]
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X, y)
    logging.info(f'Best parameters found: {grid_search.best_params_}')
    logging.info(f'Best cross-validation score: {grid_search.best_score_}')
    best_pipeline = grid_search.best_estimator_
    joblib.dump(best_pipeline, 'models/text_classification_pipeline.pkl')
    logging.info('Model training complete and saved.')
    return best_pipeline

def privacy_adapter(cleaned_text, noise_level=0.1):
    tokens = cleaned_text.split()
    num_noise_words = max(1, int(len(tokens) * noise_level))
    english_words = words.words()
    noise_words = random.choices(english_words, k=num_noise_words)
    for noise_word in noise_words:
        insert_position = random.randint(0, len(tokens))
        tokens.insert(insert_position, noise_word)
    noisy_text = ' '.join(tokens)
    return noisy_text

def classify_text(pipeline, text):
    cleaned_text = preprocess_text(text)
    noisy_text = privacy_adapter(cleaned_text)
    prediction_text = pipeline.predict([text])
    prediction_cleaned_text = pipeline.predict([cleaned_text])
    prediction_noisy_text = pipeline.predict([noisy_text])
    class_mapping = {0: 'hate_speech', 1: 'offensive_language', 2: 'neither'}
    return class_mapping[prediction_text[0]], class_mapping[prediction_cleaned_text[0]], class_mapping[prediction_noisy_text[0]]

def audio_to_text(file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
            return recognizer.recognize_google(audio, language='en-US')
    except sr.UnknownValueError:
        logging.error("Google Speech Recognition could not understand audio")
        return None
    except sr.RequestError as e:
        logging.error(f"Could not request results from Google Speech Recognition service; {e}")
        return None
    except Exception as e:
        logging.error(f"Error processing the audio file: {e}")
        return None
    
def process_results(output, filename, result_text, result_clean, result_noise):
    classification = '_'.join(filename.replace('.wav', '').split('_')[1:])
    if classification not in output:
        output[classification] = {
            'classification_count': 0,
            'raw_count': 0,
            'clean_count': 0,
            'noise_count': 0,
            'failed_count': 0
        }
    output[classification]['classification_count'] += 1
    if result_text in classification:
        output[classification]['raw_count'] += 1
    if result_clean in classification:
        output[classification]['clean_count'] += 1
    if result_noise in classification:
        output[classification]['noise_count'] += 1
    if result_noise in 'audio_to_text_failed':
        output[classification]['failed_count'] += 1

def write_to_csv(output):
    with open('results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['classification', 'classification_count', 'raw_count', 'clean_count', 'noise_count', 'failed_count'])
        for classification in output.keys():
            writer.writerow([classification, 
                             output[classification]['classification_count'], 
                             output[classification]['raw_count'], 
                             output[classification]['clean_count'], 
                             output[classification]['noise_count'],
                             output[classification]['failed_count']])

def main():
    model_path = 'models/text_classification_pipeline.pkl'
    if os.path.exists(model_path):
        pipeline = joblib.load(model_path)
        logging.info('Loaded existing model.')
    else:
        pipeline = train_model()
    
    folder_path = "data/audio"
    output = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            text = audio_to_text(file_path)
            if text:
                result_text, result_clean, result_noise = classify_text(pipeline, text)
                process_results(output, filename, result_text, result_clean, result_noise)
            else:
                process_results(output, filename, "audio_to_text_failed", "audio_to_text_failed", "audio_to_text_failed")
    write_to_csv(output)

if __name__ == "__main__":
    main()

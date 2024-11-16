import os
import logging
import pandas as pd
import re
import nltk
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
import spacy
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
nlp = spacy.load('en_core_web_sm')

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

PII_PATTERNS = {
    'Email Address': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    'Phone Number': re.compile(r'\b\d{10,15}\b'),
    'Social Security Number': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    'Credit Card Number': re.compile(r'\b(?:\d[ -]*?){13,16}\b'),
}

def read_data(file_path):
    try:
        df = pd.read_csv(file_path)        
        return list(zip(df['tweet'], df['class'], df['pii_class']))
    except Exception as e:
        logging.error(f"Error reading the file: {e}")
        return None

def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text).lower()
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in nltk.corpus.stopwords.words('english')]
    return ' '.join(tokens)

def predict_pii(text):
    for label, pattern in PII_PATTERNS.items():
        if pattern.search(text):
            return True, label
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'GPE', 'ORG', 'DATE']:
            return True, f'{ent.label_}: {ent.text}'
    
    return False, label_pii(2)

def train_model():
    df = pd.read_csv('data/labeled_data.csv')
    df['cleaned_tweet'] = df['tweet'].apply(preprocess_text)
    X, y = df['cleaned_tweet'], df['class']

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

def label_hate(classification):
    class_mapping = {0: 'hate_speech', 1: 'offensive_language', 2: 'neither'}
    return class_mapping[classification]

def label_pii(classification):
    class_mapping = {0: 'not_classified', 1: 'no_pii', 2: 'contains_pii'}
    return class_mapping[classification]

def classify_text(pipeline, text, hate_class, pii_class, counts):
    cleaned_text = preprocess_text(text)

    if pii_class != 0:
        predicted_contains_pii, predicted_pii_type = predict_pii(cleaned_text)
        predicted_pii_class = 2 if predicted_contains_pii else 1
        predicted_pii_label = label_pii(predicted_pii_class)
        actual_pii_label = label_pii(pii_class)

        counts['correct_pii'] += (predicted_pii_label == actual_pii_label)
        counts['wrong_pii'] += (predicted_pii_label != actual_pii_label)
        counts['predict_pii'] += (predicted_pii_label == 'contains_pii')
        counts['actual_pii'] += (actual_pii_label == 'contains_pii')

    prediction = pipeline.predict([cleaned_text])
    predicted_hate_class = prediction[0]
    predicted_hate_label = label_hate(predicted_hate_class)
    actual_hate_label = label_hate(hate_class)

    counts['correct_hate'] += (predicted_hate_label == actual_hate_label)
    counts['wrong_hate'] += (predicted_hate_label != actual_hate_label)
    counts['predict_hate'] += (predicted_hate_label != 'neither')
    counts['actual_hate'] += (actual_hate_label != 'neither')

def create_histogram_acc(counts):
    correct_pii = counts['correct_pii']
    wrong_pii = counts['wrong_pii']
    correct_hate = counts['correct_hate']
    wrong_hate = counts['wrong_hate']
    barWidth = 0.5

    total_pii = correct_pii + wrong_pii
    total_hate = correct_hate + wrong_hate
    pii_accuracy = (correct_pii / total_pii * 100)
    hate_accuracy = (correct_hate / total_hate * 100)

    labels = [f'Contains PII \n ({pii_accuracy:.0f}%)', 
              f'Contains Hate \n ({hate_accuracy:.0f}%)']
    accuracies = [pii_accuracy, hate_accuracy]

    x_positions = np.arange(len(labels))

    fig, ax = plt.subplots(dpi=120, figsize=(8, 6))

    ax.bar(x_positions, accuracies, width=barWidth, color='#89CFF0', label='Accuracy (%)')

    ax.set_xlabel('Classifications')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def create_histogram_count(counts):
    predict_pii_count = counts['predict_pii']
    actual_pii_count = counts['actual_pii']
    predict_hate_count = counts['predict_hate']
    actual_hate_count = counts['actual_hate']
    barWidth = 0.4

    labels = ['Contains PII', 'Contains Hate']
    predict_values = [predict_pii_count, predict_hate_count]
    actual_values = [actual_pii_count, actual_hate_count]

    x_positions = np.arange(len(labels))
    r1 = x_positions - barWidth / 2
    r2 = x_positions + barWidth / 2

    fig, ax = plt.subplots(dpi=120, figsize=(8, 6))

    ax.bar(r1, predict_values, width=barWidth, color='#89CFF0', label='Prediction')
    ax.bar(r2, actual_values, width=barWidth, color='#2d7f5e', label='Actual')

    ax.set_xlabel('Classifications')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('Counts')
    ax.legend(loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def main():
    model_path = 'models/text_classification_pipeline.pkl'
    pipeline = joblib.load(model_path) if os.path.exists(model_path) else train_model()
    
    filename = "data/labeled_data.csv"
    data = read_data(filename)

    counts = {
        'correct_pii': 0,
        'wrong_pii': 0,
        'correct_hate': 0,
        'wrong_hate': 0,
        'predict_pii': 0,
        'actual_pii': 0,
        'predict_hate': 0,
        'actual_hate': 0,
    }
    
    if data:
        for text, hate_class, pii_class in data:
            classify_text(pipeline, text, hate_class, pii_class, counts)
        create_histogram_acc(counts)
        create_histogram_count(counts)   

if __name__ == "__main__":
    main()

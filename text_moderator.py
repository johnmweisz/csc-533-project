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

# Setup logging and load necessary models
logging.basicConfig(level=logging.INFO)
nlp = spacy.load('en_core_web_sm')

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Regex patterns for PII detection
PII_PATTERNS = {
    'Email Address': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    'Phone Number': re.compile(r'\b\d{10,15}\b'),
    'Social Security Number': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    'Credit Card Number': re.compile(r'\b(?:\d[ -]*?){13,16}\b'),
}

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

def classify_text(pipeline, text, hate_class, pii_class, predict_pii_class_counts, actual_pii_class_counts, predict_hate_class_counts, actual_hate_class_counts):
    cleaned_text = preprocess_text(text)

    if pii_class != 0:
        predicted_contains_pii, predicted_pii_type = predict_pii(cleaned_text)
        predict_pii_class_counts['contains_pii' if predicted_contains_pii else 'no_pii'] += 1
        actual_pii_class_counts['contains_pii' if pii_class == 2 else 'no_pii'] += 1

    prediction = pipeline.predict([cleaned_text])
    predicted_hate_class = prediction[0]
    predict_hate_class_counts['no_hate' if predicted_hate_class != 2 else 'contains_hate'] += 1
    actual_hate_class_counts['no_hate' if hate_class != 2 else 'contains_hate'] += 1

def read_data(file_path):
    try:
        df = pd.read_csv(file_path)        
        return list(zip(df['tweet'], df['class'], df['pii_class']))
    except Exception as e:
        logging.error(f"Error reading the file: {e}")
        return None

def create_histogram(predict_pii_class_counts, actual_pii_class_counts, predict_hate_class_counts, actual_hate_class_counts):
    barWidth = 0.2 # Width of bars in the histogram
    predict_bar = predict_pii_class_counts.values() # Counts for PII Classification Predictions
    actual_bar = actual_pii_class_counts.values() # Counts for actual PII Classification

    r1 = np.arange(len(predict_bar)) # Range for x-axis
    r2 = r1 + barWidth

    # Create the figure for the histogram plot
    fig, ax = plt.subplots(dpi=120)

    # Plot the bars for prediction and actual counts
    ax.bar(r1, predict_bar, width=barWidth, color='#89CFF0', label='Prediction')
    ax.bar(r2, actual_bar, width=barWidth, color='#2d7f5e', label='Actual')

    # Set the labels and ticks for the x-axis
    ax.set_xlabel('PII Classification')
    ax.set_xticks(r1 + .5*barWidth)

    # Labels for x-axis
    ax.set_xticklabels(actual_pii_class_counts.keys())

    ax.legend(bbox_to_anchor=(1.0, 1.0)) # Add a legend to the plot

    plt.show() # Display the plot

def main():
    model_path = 'models/text_classification_pipeline.pkl'
    pipeline = joblib.load(model_path) if os.path.exists(model_path) else train_model()
    
    filename = "data/labeled_data.csv"
    data = read_data(filename)
    predict_pii_class_counts = {'no_pii': 0, 'contains_pii': 0}
    actual_pii_class_counts = {'no_pii': 0, 'contains_pii': 0}
    predict_hate_class_counts = {'no_hate': 0, 'contains_hate': 0}
    actual_hate_class_counts = {'no_hate': 0, 'contains_hate': 0}
    if data:
        for text, hate_class, pii_class in data:
            classify_text(pipeline, text, hate_class, pii_class, predict_pii_class_counts, actual_pii_class_counts, predict_hate_class_counts, actual_hate_class_counts)
    create_histogram(predict_pii_class_counts, actual_pii_class_counts, predict_hate_class_counts, actual_hate_class_counts)

if __name__ == "__main__":
    main()

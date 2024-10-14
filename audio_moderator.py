import os
import logging
import speech_recognition as sr
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)

# Ensure NLTK data is downloaded
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def preprocess_text(text):
    # Remove URLs, mentions, hashtags, and special characters
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize and lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    
    return ' '.join(tokens)

def train_model():
    df = pd.read_csv('data/labeled_data.csv')
    df['cleaned_tweet'] = df['tweet'].apply(preprocess_text)
    
    X = df['cleaned_tweet']
    y = df['class']
    
    # Handle class imbalance
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(zip(np.unique(y), class_weights))
    
    # Create a pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000, class_weight=class_weight_dict))
    ])
    
    # Cross-validation
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1_weighted')
    logging.info(f'Cross-validation F1 scores: {scores}')
    logging.info(f'Mean F1 score: {scores.mean()}')
    
    # Fit the model
    pipeline.fit(X, y)
    
    # Save the model
    joblib.dump(pipeline, 'models/text_classification_pipeline.pkl')
    logging.info('Model training complete and saved.')
    
    return pipeline

def classify_text(pipeline, text):
    cleaned_text = preprocess_text(text)
    prediction = pipeline.predict([cleaned_text])
    class_mapping = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
    return class_mapping[prediction[0]]

def audio_to_text(file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
            # You can specify language parameters if needed
            return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        logging.error("Google Speech Recognition could not understand audio")
        return None
    except sr.RequestError as e:
        logging.error(f"Could not request results from Google Speech Recognition service; {e}")
        return None
    except Exception as e:
        logging.error(f"Error processing the audio file: {e}")
        return None

def main():
    # Check if the model exists to avoid retraining
    model_path = 'models/text_classification_pipeline.pkl'
    if os.path.exists(model_path):
        pipeline = joblib.load(model_path)
        logging.info('Loaded existing model.')
    else:
        pipeline = train_model()
    
    folder_path = "data/audio"
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            text = audio_to_text(file_path)
            if text:
                result = classify_text(pipeline, text)
                print(f"{filename} classified as: {result}")
            else:
                print(f"{filename} could not be processed.")

if __name__ == "__main__":
    main()

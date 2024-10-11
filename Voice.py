import os
import speech_recognition as sr
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import nltk
import pandas as pd
import re

nltk.download('stopwords')

#Load dataset using pandas
df = pd.read_csv('data/offensive_sentences.csv')

#Fix text from csv
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

df['cleaned_tweet'] = df['tweet'].apply(preprocess_text)

#vectorization stuff
vectorizer = TfidfVectorizer(max_features=5000, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(df['cleaned_tweet']).toarray()
y = df['class']

#Split the data to train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the model, increase max iterations so it won't error out
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


#Function used to classify a piece of text
def classify_text(text):
    cleaned_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([cleaned_text]).toarray()
    prediction = model.predict(text_vectorized)
    class_mapping = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
    return class_mapping[prediction[0]]

#Function used to convert a .wav to text
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

    #Simple selection for user. 1 = input text, 2 = input file path for a .wav
    choice = input("Enter 1 to classify a sentence, or 2 to classify an audio file: ")

    if choice == '1':
        user_text = input("Enter a sentence to classify: ")
        result = classify_text(user_text)
        print(f"The text is classified as: {result}")

    elif choice == '2':

        file_path = input("Enter the path to an audio file: ")

        if not os.path.exists(file_path):
            print("File not found!")
            return

        #Convert the .wav
        text = audio_to_text(file_path)
        if text is not None:

            ### DEBUG:
            #print(f"Transcribed Text: {text}")
            result = classify_text(text)
            print(f"The audio file is classified as: {result}")
        else:
            print("Could not process the audio file.")

    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()

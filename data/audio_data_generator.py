import pandas as pd
from gtts import gTTS

def text_to_audio(text, filename='output_audio.mp3', lang='en'):
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(filename)
    print(f"Audio file saved as {filename}")

def get_highest_category(row):
    categories = {
        'hate_speech': row['hate_speech'],
        'offensive_language': row['offensive_language'],
        'neither': row['neither']
    }
    highest_category = max(categories, key=categories.get)
    return highest_category

def csv_to_audio(csv_file, dir, lang='en'):
    try:
        data = pd.read_csv(csv_file)

        required_columns = ['count', 'hate_speech', 'offensive_language', 'neither', 'class', 'tweet']
        if not all(col in data.columns for col in required_columns):
            print(f"One or more required columns {required_columns} not found in the CSV file.")
            return

        for idx, row in data.iterrows():
            classification = get_highest_category(row)
            output_filename = f"{dir}/{idx}_{classification}.wav"
            tweet_text = row['tweet']
            text_to_audio(tweet_text, output_filename, lang)

    except FileNotFoundError:
        print(f"File '{csv_file}' not found.")

# Input CSV file and column with text
csv_file = 'labeled_data.csv'

# Call the function to convert CSV data to audio files
csv_to_audio(csv_file, dir='audio', lang='en')

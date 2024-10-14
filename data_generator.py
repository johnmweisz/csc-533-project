import pandas as pd
from gtts import gTTS
import re
from io import BytesIO
from pydub import AudioSegment

def text_to_audio(text, filename='output_audio.wav', lang='en'):
    tts = gTTS(text=text, lang=lang, slow=False)
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    sound = AudioSegment.from_file(mp3_fp, format='mp3')
    sound.export(filename, format='wav')
    print(f"Audio file saved as {filename}")

def sanitize_text(text):
    text = re.sub(r"http\S+|@\S+|#\S+|[^A-Za-z\s]", "", text)
    return text.split(':')[-1].strip()

def get_classification(row):
    class_mapping = {0: 'hate_speech', 1: 'offensive_language', 2: 'neither'}
    return class_mapping[row['class']]

def csv_to_audio(csv_file, dir, lang='en', num_samples=100):
    data = pd.read_csv(csv_file)

    required_columns = ['class', 'tweet']
    if not all(col in data.columns for col in required_columns):
        print(f"One or more required columns {required_columns} not found in the CSV file.")
        return

    if num_samples is not None:
        data = data.sample(n=num_samples)

    for idx, row in data.iterrows():
        classification = get_classification(row)
        output_filename = f"{dir}/{idx}_{classification}.wav"
        tweet_text = sanitize_text(row['tweet'])
        text_to_audio(tweet_text, output_filename, lang)

if __name__ == '__main__':
    csv_to_audio('data/labeled_data.csv', dir='data/audio', lang='en')

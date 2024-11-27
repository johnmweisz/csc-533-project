import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping

filename = "data/labeled_data.csv"
df = pd.read_csv(filename)

X = df['tweet']
y = df['pii_class']

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
X_tokenized = tokenizer.texts_to_sequences(X)

max_length = 100
X_padded = pad_sequences(X_tokenized, maxlen=max_length, padding='post', truncating='post')

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_padded, y_encoded, test_size=0.3, random_state=42)

model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=max_length),
    SpatialDropout1D(0.3),
    Bidirectional(LSTM(150, dropout=0.3, recurrent_dropout=0.3)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=1)

y_pred = (model.predict(X_padded) > 0.5).astype("int32")

df['pii_prediction_lstm'] = y_pred

output_path = 'data/labeled_data_with_lstm_pii_predictions.csv'
df.to_csv(output_path, index=False)

print(f"Updated CSV saved to: {output_path}")

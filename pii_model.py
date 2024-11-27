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
from sklearn.utils import resample

# Load the training CSV file
training_file_path = 'data/balanced_pii_training_data.csv'
df_training = pd.read_csv(training_file_path)

# Prepare the training data
X_train_data = df_training['message']
y_train_data = df_training['pii_class']

# Tokenize the training text data
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_data)
X_train_tokenized = tokenizer.texts_to_sequences(X_train_data)

# Pad the sequences to ensure uniform input length
max_length = 100
X_train_padded = pad_sequences(X_train_tokenized, maxlen=max_length, padding='post', truncating='post')

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_data)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_padded, y_train_encoded, test_size=0.3, random_state=42)

# Define LSTM model with Bidirectional LSTM
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=max_length),
    SpatialDropout1D(0.3),
    Bidirectional(LSTM(150, dropout=0.3, recurrent_dropout=0.3)),
    Dense(1, activation='sigmoid')
])

# Compile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with more epochs and early stopping
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)

# Load the actual data CSV file to make predictions
actual_data_file_path = 'data/labeled_data.csv'
df_actual = pd.read_csv(actual_data_file_path)

# Prepare the actual data for prediction
X_actual = df_actual['tweet']
X_actual_tokenized = tokenizer.texts_to_sequences(X_actual)
X_actual_padded = pad_sequences(X_actual_tokenized, maxlen=max_length, padding='post', truncating='post')

# Predict on the actual dataset
y_pred = (model.predict(X_actual_padded) > 0.5).astype("int32") + 1

# Add LSTM predictions to the DataFrame
df_actual['pii_prediction_lstm'] = y_pred

# Save the updated DataFrame with LSTM predictions to a new CSV file
output_path = 'data/labeled_data_with_lstm_pii_predictions.csv'
df_actual.to_csv(output_path, index=False)

print(f"Updated CSV saved to: {output_path}")

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# 1. Muat Data
df = pd.read_csv('data.csv')
df = df.dropna()

texts = df['text'].astype(str).values
labels = df['label'].values

# 2. Parameter (Wajib sama dengan app.py)
vocab_size = 5000
max_length = 50
embedding_dim = 32

# 3. Pra-pemrosesan Teks
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# 4. Pembagian Data
X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2, random_state=42)

# 5. Arsitektur Model (Sama persis dengan struktur di app.py)
# --- UPDATE LANGKAH 5 PADA TRAIN.PY ---
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length, name="embed_layer"),
    LSTM(64, name="lstm_layer"),
    Dropout(0.5, name="dropout_layer"),
    Dense(1, activation='sigmoid', name="dense_layer")
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 6. Proses Pelatihan
model.fit(X_train, y_train, epochs=10, validation_split=0.1, verbose=1)

# 7. SIMPAN BOBOT DAN TOKENIZER (Solusi Anti-Error Versi)
# --- UPDATE LANGKAH 7 PADA TRAIN.PY ---
# Gunakan save_weights format biner Keras dengan nama layer eksplisit
model.save_weights('model_lstm_fixed.weights.h5')

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Sukses: File 'model_lstm_fixed.weights.h5' telah dibuat!")
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# 1. Konfigurasi Halaman Utama
st.set_page_config(page_title="Sentimen Analisis LSTM Streamlit", page_icon="📊", layout="wide")

# 2. Fungsi Load Resource yang Selaras
@st.cache_resource
def load_resources():
    # Bangun arsitektur dengan nama layer yang persis sama dengan train.py
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=5000, output_dim=32, input_length=50, name="embed_layer"),
        tf.keras.layers.LSTM(64, name="lstm_layer"),
        tf.keras.layers.Dropout(0.5, name="dropout_layer"),
        tf.keras.layers.Dense(1, activation='sigmoid', name="dense_layer")
    ])
    
    # ---> TAMBAHKAN BARIS INI UNTUK MENGATASI ERROR "NOT YET BUILT" <---
    # Memaksa model menginisiasi memori dengan ekspektasi panjang teks = 50
    model.build(input_shape=(None, 50)) 
    
    # Memuat bobot murni lewat framework biner resmi
    model.load_weights('model_lstm_fixed.weights.h5')
        
    # Memuat tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    return model, tokenizer

# 3. Komponen UI: Sidebar (Informasi Sistem)
with st.sidebar:
    st.header("Informasi Sistem")
    st.markdown("""
    * **Arsitektur:** Long Short-Term Memory (LSTM)
    * **Tugas:** Text Mining
    """)
    st.divider()
    st.caption("Masukkan teks pada area utama untuk melihat prediksi sentimen biner (0: Negatif, 1: Positif).")

# 4. Komponen UI: Panel Utama
st.title("Aplikasi Analisis Sentimen")
st.markdown("### Deteksi sentimen teks ulasan menggunakan Deep Learning")

try:
    model, tokenizer = load_resources()
    
    # Form Input Teks
    with st.form(key='sentiment_form'):
        user_input = st.text_area("Masukkan Teks Ulasan:", placeholder="Ketik atau paste ulasan Anda di sini...", height=150)
        submit_button = st.form_submit_button(label='Analisis Sentimen')
        
    # Logika Pemrosesan Model
    if submit_button:
        if not user_input.strip():
            st.warning("Teks ulasan kosong, masukkan teks terlebih dahulu!")
        else:
            # Preprocessing teks input
            max_length = 50
            sequence = tokenizer.texts_to_sequences([user_input])
            padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
            
            # Prediksi nilai Sigmoid
            prediction = model.predict(padded_sequence)
            score = float(prediction[0][0])
            
            # Output Layout Hasil Analisis
            st.markdown("#### Hasil Analisis")
            col1, col2 = st.columns(2)
            
            with col1:
                if score >= 0.5:
                    st.success("Sentimen: **POSITIF**")
                else:
                    st.error("Sentimen: **NEGATIF**")
                    
            with col2:
                st.metric(label="Skor Probabilitas (Sigmoid)", value=f"{score:.4f}")
                st.progress(score, text=f"Tingkat Keyakinan: {score * 100:.2f}%")

except Exception as e:
    st.error(f"Gagal memuat model atau tokenizer: {e}")
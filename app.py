import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd

# Konfigurasi halaman utama
st.set_page_config(page_title="Sentimen Analisis LSTM", page_icon="📊", layout="wide")

# Fungsi pemuatan model dalam cache
@st.cache_resource
def load_model_data():
    # Di dalam fungsi load_model_data()
    model = tf.keras.models.load_model('model_lstm.keras')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

try:
    model, tokenizer = load_model_data()
    model_loaded = True
except Exception as e:
    st.error(f"Gagal memuat model atau tokenizer: {e}. Pastikan file model_lstm.h5 dan tokenizer.pickle berada di direktori yang sama.")
    model_loaded = False

# Tata letak Sidebar
with st.sidebar:
    st.header("Informasi Sistem")
    st.write("**Arsitektur:** Long Short-Term Memory (LSTM)")
    st.write("**Tugas:** Text Mining")
    st.divider()
    st.write("Masukkan teks pada area utama untuk melihat prediksi sentimen biner (0: Negatif, 1: Positif).")

# Tata letak Utama
st.title("Aplikasi Analisis Sentimen")
st.markdown("Deteksi sentimen teks ulasan menggunakan *Deep Learning*.")

if model_loaded:
    # Form input untuk efisiensi eksekusi
    with st.form(key='nlp_form'):
        user_input = st.text_area("Masukkan Teks Ulasan:", placeholder="Contoh: Makanannya enak dan pelayanannya cepat...", height=150)
        submit_button = st.form_submit_button(label='Analisis Sentimen')

    # Logika Prediksi
    if submit_button:
        if user_input.strip() == "":
            st.warning("Teks ulasan tidak boleh kosong.")
        else:
            # Pra-pemrosesan data
            max_length = 50 
            sequence = tokenizer.texts_to_sequences([user_input])
            padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
            
            # Eksekusi model
            prediction = model.predict(padded_sequence)
            score = prediction[0][0]
            
            st.markdown("### Hasil Analisis")
            col1, col2 = st.columns(2)
            
            with col1:
                if score >= 0.5:
                    st.success("Sentimen: **POSITIF**")
                else:
                    st.error("Sentimen: **NEGATIF**")
                    
            with col2:
                st.metric(label="Skor Probabilitas (Sigmoid)", value=f"{score:.4f}")
                st.progress(float(score), text="Tingkat Keyakinan Positif")
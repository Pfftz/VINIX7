import streamlit as st
import pickle
import os
import re
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    stemmer = StemmerFactory().create_stemmer()
except Exception:
    stemmer = None


def clean_text(s):
    s = str(s).lower().strip()
    s = re.sub(r'http\S+', ' ', s)
    s = re.sub(r'[^0-9a-z\s]', ' ', s)
    s = re.sub(r'\b(ga|gak|nggak|enggak|tdk)\b', ' tidak ', s)
    s = re.sub(r'(.)\1{2,}', r'\1\1', s)
    s = re.sub(r'\s+', ' ', s).strip()
    if stemmer:
        s = stemmer.stem(s)
    return s

@st.cache_resource
def load_sentiment_model():
    model_path = 'model_tokopedia.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

model = load_sentiment_model()

st.set_page_config(page_title="Sentimen Tokopedia", layout="centered")
st.title("Analisis Sentimen Ulasan Tokopedia")
st.markdown("""
Aplikasi ini menggunakan model **Logistic Regression** dengan TF-IDF word + char n-gram untuk memprediksi kategori sentimen dari teks ulasan produk Tokopedia.
""")

st.divider()

user_review = st.text_area(
    "Masukkan ulasan pelanggan di bawah ini:",
    placeholder="Contoh: Barang bagus, pengiriman sangat cepat...",
    height=150
)

if st.button("Prediksi Sentimen"):
    if model is None:
        st.error("File 'model_tokopedia.pkl' tidak ditemukan. Pastikan serialisasi model sudah berhasil.")
    elif user_review.strip() == "":
        st.warning("Mohon masukkan teks ulasan terlebih dahulu untuk melakukan prediksi.")
    else:
        txt = clean_text(user_review)
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba([txt])[0]
            classes = model.named_steps['clf'].classes_
            top_idx = probs.argmax()
            prediction = classes[top_idx]
            confidence = probs[top_idx]

            st.subheader("Hasil Analisis:")
            if prediction == "Positif":
                st.success(f"Sentimen Terdeteksi: **{prediction}** — Konfidence: {confidence:.2f}")
            elif prediction == "Negatif":
                st.error(f"Sentimen Terdeteksi: **{prediction}** — Konfidence: {confidence:.2f}")
            else:
                st.info(f"Sentimen Terdeteksi: **{prediction}** — Konfidence: {confidence:.2f}")

            if confidence < 0.60:
                st.warning("Konfidence rendah, hasil bisa kurang stabil. Pertimbangkan evaluasi manual.")

            st.write("Probabilitas per kelas:")
            for c, p in zip(classes, probs):
                st.write(f"- {c}: {p:.2f}")
        else:
            prediction = model.predict([txt])[0]
            st.subheader("Hasil Analisis:")
            if prediction == "Positif":
                st.success(f"Sentimen Terdeteksi: **{prediction}**")
            elif prediction == "Negatif":
                st.error(f"Sentimen Terdeteksi: **{prediction}**")
            else:
                st.info(f"Sentimen Terdeteksi: **{prediction}**")

st.caption("Vinix7 Project-Based Internship - Kelompok 17")

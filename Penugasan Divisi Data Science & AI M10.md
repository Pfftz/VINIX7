## Page 1

PROJECT-BASED INTERNSHIP VINIX7 
Modul 10: Deployment & MLOps 
Konteks Bisnis: Setelah berhasil membangun pipeline ETL pada Modul 9, tim Data Science 
kini ditugaskan untuk melatih model klasifikasi Machine Learning menggunakan dataset ulasan 
Tokopedia tersebut. Model ini bertujuan untuk memprediksi kategori sentimen pelanggan 
(Positif, Negatif, atau Netral) secara otomatis. Klien meminta hasil model ini didemonstrasikan 
dalam bentuk dashboard interaktif agar tim marketing dapat melakukan pengujian secara 
mandiri. Anda diminta untuk mengintegrasikan proses modeling, serialisasi, hingga deployment 
sederhana di lingkungan cloud. 
Instruksi Umum: 
1.​ Penugasan dikerjakan menggunakan Google Colab. 
2.​ Implementasi antarmuka menggunakan library Streamlit/Flask ataupun library lainnya. 
3.​ Tambahkan sel teks (Markdown) untuk setiap interpretasi atau jawaban teoretis. 
Tahap 1: Persiapan Data & Serialisasi Model 
1.​ Unduh dataset ulasan Tokopedia dari Kaggle: 
salmanabdu/tokopedia-product-reviews-2025. 
2.​ Lakukan pra-pemrosesan: bersihkan nilai kosong, buat kolom label Sentimen 
berdasarkan Rating (Rating > 3: Positif, < 3: Negatif, = 3: Netral). 
3.​ (Opsional) Ambil sampel data (misal: 5.000 atau 10.000 baris) agar proses training di 
Colab berjalan lebih cepat. 
4.​ Latih model klasifikasi teks (misal: Multinomial Naive Bayes) menggunakan pipeline 
Scikit-Learn/Pytorch/Tensorflow. 
5.​ Simpan objek model ke dalam file biner model_tokopedia.pkl menggunakan 
pickle ataupun menyimpan model ke file format lainnya. 
Tahap 2: Pengembangan Dashboard Streamlit 
1.​ Gunakan magic command %%writefile app.py di Colab untuk membuat skrip 
Streamlit/Flask ataupun library lainnya. 
2.​ Rancang UI yang memiliki fitur: 
○​ Memuat file model_tokopedia.pkl. 
○​ Menampilkan judul aplikasi (misal: "Analisis Sentimen Ulasan Tokopedia"). 
○​ Kotak teks input ulasan percobaan. 
○​ Tombol prediksi dan visualisasi hasil prediksi . 
Tahap 3: Deployment & Tunneling 
1.​ Jalankan aplikasi Streamlit di latar belakang sistem Colab.

---

## Page 2

2.​ Gunakan localtunnel atau cloudflared ataupun pendekatan lainnya untuk 
membuat URL publik yang dapat diakses dari luar. 
3.​ Lampirkan screenshot dashboard yang telah berhasil berjalan. 
Tahap 4: Pemahaman MLOps & Containerization  
Jawablah pertanyaan berikut pada sel Markdown: 
1.​ Docker: Tuliskan draf Dockerfile sederhana untuk mengemas aplikasi Streamlit ini. 
2.​ Monitoring (Data Drift): Jika akurasi prediksi model sentimen Tokopedia ini menurun 
drastis tahun depan padahal tidak ada kode yang diubah, fenomena apa ini dan apa 
solusinya dalam kerangka MLOps? 
3.​ CI/CD: Jelaskan bagaimana otomatisasi CI/CD mempermudah tim untuk mengganti 
versi model di server tanpa membuat dashboard mengalami downtime. 
KRITERIA PENILAIAN 
1.​ Serialisasi Model (25%) Maksimal: Peserta sukses mengunduh data Tokopedia, 
memproses label sentimen, melatih model, dan menghasilkan file 
model_tokopedia.pkl ataupun model yang di save dengan pendekatan lain tanpa 
error. 
2.​ Implementasi UI [Streamlit/Flask] (35%) Maksimal: Skrip app.py berhasil dibangun 
dengan fungsionalitas memanggil model yang benar, serta Cloudflare/local ataupun 
tunnel lainnya dan sukses menampilkan UI yang berjalan lancar (dibuktikan melalui hasil 
screenshot). 
3.​ Desain Dockerfile (20%) Maksimal: Struktur perintah Dockerfile yang dituliskan di 
jawaban Markdown logis, mencakup instalasi requirements, penyalinan model, dan 
perintah eksekusi Streamlit. 
4.​ Analisis MLOps (20%) Maksimal: Argumen mengenai penanganan data drift (melalui 
monitoring dan retraining) serta konsep CI/CD (zero downtime) terstruktur dan 
menjawab masalah industri riil.

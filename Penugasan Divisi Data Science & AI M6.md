## Page 1

PROJECT-BASED INTERNSHIP VINIX7 
Modul 6: Supervised Learning - Feature Engineering & Penanganan Data Imbalance 
Konteks Bisnis: Klien brand aggregator Tokopedia kita menuntut model Machine Learning 
yang sesungguhnya untuk mengotomatisasi deteksi komplain pelanggan. Jika di modul 
sebelumnya Anda hanya mengevaluasi model statis, kini Anda bertugas membangun model 
klasifikasi dari nol menggunakan data mentah. 
Tantangan utama Anda adalah mengolah berbagai tipe data (teks, numerik, kategorikal) dan 
mengatasi tantangan distribusi target pada data e-commerce. Klien ingin model yang sangat 
sensitif terhadap komplain (sentimen negatif) agar tim Customer Service (CS) dapat merespons 
keluhan dengan sigap sebelum pelanggan pindah ke kompetitor (churn). 
Instruksi Umum: 
●​ Pengerjaan dilakukan menggunakan Jupyter Notebook atau Google Colab. 
●​ Setiap selesai menjalankan tahap utama, Anda wajib menambahkan sel Markdown 
untuk menuliskan interpretasi teknis dan dampaknya terhadap bisnis. 
Tahap 1: Data Cleansing & Handling 'Neutral' 
1.​ Load dataset tokopedia_product_reviews_2025.csv menggunakan Pandas. 
2.​ Hapus baris data yang memiliki label sentimen 'neutral'. 
3.​ Buang (drop) kolom yang berpotensi menyebabkan data leakage atau tidak relevan: 
review_date, review_id, product_name, product_url, product_id, shop_id, 
dan rating. 
Tahap 2: Feature Engineering & Preprocessing 
1.​ Ekstrak fitur baru bernama review_length yang berisi jumlah karakter teks dari kolom 
review_text. 
2.​ Ubah target target 'positive' menjadi 1 dan 'negative' menjadi 0. Pisahkan 
features (X) dan target (y). 
3.​ Lakukan train_test_split (80% Train, 20% Test) dengan random_state=42. 
Penting: Lakukan ini sebelum tahap transformasi untuk mencegah Data Leakage. 
4.​ Gunakan ColumnTransformer untuk melakukan Fit & Transform pada data Train, lalu 
hanya Transform pada data Test dengan skema: 
○​ review_text: Gunakan TfidfVectorizer (batasi max_features=1000). 
○​ product_category & product_variant: Gunakan One-Hot Encoding. 
○​ product_price, sold_count, review_length: Gunakan 
StandardScaler.

---

## Page 2

Tahap 3: Eksperimen Model (Baseline) 
1.​ Latih model XGBoost Classifier atau Random Forest Classifier atau model 
klasifikasi lainnya menggunakan data Train yang sudah diproses. Biarkan seluruh 
parameter secara default. 
2.​ Lakukan prediksi pada data Test dan tampilkan classification_report. 
3.​ Interpretasi 1: Analisis hasil metrik Anda. Bagaimana nilai metrik evaluasi (terutama 
Recall) untuk kelas 0 (negatif)? Secara teknis, mengapa model menghasilkan pola 
metrik seperti itu? Apa bahayanya bagi operasional CS jika model baseline ini 
di-deploy? 
Tahap 4: Eksperimen Penanganan Imbalance & Tuning Tugas Anda sekarang adalah 
memodifikasi model agar mampu mengenali komplain (kelas 0) secara optimal. Anda 
dibebaskan menggunakan pendekatan penanganan imbalance data. 
1.​ Pilih minimal satu pendekatan berikut dan ujikan bersama GridSearchCV (cv=3 atau 
5): 
○​ Menggunakan parameter pembobotan bawaan algoritma (misal: 
scale_pos_weight pada XGBoost atau class_weight='balanced' pada 
Tree/SVC). 
○​ Menggunakan teknik Undersampling (misal: RandomUnderSampler dari library 
imblearn). 
○​ Menggunakan teknik Oversampling (misal: SMOTE dari library imblearn). 
(Catatan: Jika Anda menggunakan teknik resampling, Anda wajib menggunakan 
Pipeline dari library imblearn (bukan dari sklearn) agar tidak terjadi Data 
Leakage saat proses Cross-Validation internal). 
○​ Optimalisasi dan pendekatan lainnya. 
2.​ Latih model hasil eksperimen Anda, lalu uji pada data Test. Tampilkan 
classification_report terbarunya. 
3.​ Interpretasi 2 (Analisis Kondisi Model): 
○​ Pendekatan penyeimbangan apa yang Anda gunakan? 
○​ Evaluasi perubahan performa model. Jika terjadi perubahan signifikan atau 
trade-off (tarik-ulur) yang ekstrem antara nilai Recall dan Precision pada kelas 0 
setelah Anda menangani imbalance, jelaskan mengapa secara teknis algoritma 
memberikan keluaran/tebakan seperti itu. 
Tahap 5: Kesimpulan Akhir & Keputusan Bisnis  
Interpretasi 3: Apapun hasil performa metrik akhir model Anda di Tahap 4 (meskipun belum 
sempurna atau tidak seimbang), bertindaklah sebagai Data Scientist yang harus memberikan 
rekomendasi final kepada manajemen. Apakah model eksperimen Anda di Tahap 4 lebih layak 
digunakan sebagai Sistem Triase (Prioritas) Tiket CS dibandingkan model Baseline? 
Gunakan argumen komparasi estimasi beban kerja CS (menimbang risiko penanganan False 
Positive vs False Negative) untuk membenarkan argumen teknis Anda.

---

## Page 3

KRITERIA PENILAIAN 
1. Data Preparation & Preprocessing (20%) 
●​ Maksimal: Peserta secara tepat membersihkan data dari variabel noise dan leakage, 
meletakkan proses data splitting di urutan yang tepat sebelum manipulasi skala, dan 
berhasil merangkai transformasi berbagai tipe data (text, num, cat) menggunakan skema 
pemrograman yang efisien (seperti ColumnTransformer). 
2. Implementasi Eksperimen & Validasi (30%) 
●​ Maksimal: Peserta mengimplementasikan teknik penanganan ketidakseimbangan kelas 
secara terprogram. Peserta menerapkan Cross-Validation (CV) dengan kaidah yang 
benar. Jika menggunakan metode resampling (Under/Oversampling), peserta berhasil 
menggunakan arsitektur Pipeline khusus untuk mencegah terjadinya kebocoran data 
(data leakage) ke dalam validation fold selama proses CV. 
3. Ketajaman Interpretasi Teknis (Tahap 3 & Tahap 4) (30%) 
●​ Maksimal: Peserta mampu mengidentifikasi masalah utama pada model baseline. Pada 
Tahap 4, peserta tidak sekadar membaca angka metrik, melainkan mampu menguraikan 
fenomena analitik yang terjadi (khususnya relasi/ trade-off metrik kelas minoritas) yang 
merupakan efek samping logis dari paksaan algoritma dalam mengenali pola data yang 
telah direkayasa distribusinya. 
4. Kualitas Keputusan Bisnis (Tahap 5) (20%) 
●​ Maksimal: Peserta mampu menerjemahkan keluaran matematis algoritma ke dalam 
narasi bahasa operasional dan finansial. Argumen kelayakan deployment didukung oleh 
perbandingan simulasi beban kerja (seperti waktu/sumber daya yang terbuang) dan 
prioritas pencegahan kehilangan pelanggan (churn) secara logis dan realistis.

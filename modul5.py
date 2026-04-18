# %% [markdown]
# # Penugasan Divisi Data Science & AI M5
# ## Evaluasi Model Machine Learning untuk Klasifikasi Sentimen Ulasan Tokopedia
#
# Notebook ini melanjutkan progress sebelumnya dan sudah dilengkapi dengan interpretasi pada setiap tahap sesuai instruksi penugasan.

# %%
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from google.colab import files
uploaded = files.upload()

# %%
!apt-get install unrar - y
!unrar x - o + "Dataset Tugas Data Science M5.rar"
!ls

# %%

# %%
# Tahap 1.1: Load dataset menggunakan Pandas
df = pd.read_csv('tokopedia_product_reviews_2025.csv')

# Tampilkan data awal
display(df.head())

# Tahap 1.2: Isolasi kolom teks ulasan (X) dan label sentimen (y)
X = df['review_text']
y = df['sentiment_label']

# Tahap 1.3: Membagi data menjadi Training (80%) dan Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

# Tahap 1.4: Menampilkan dimensi (shape) dari data Training dan Testing
print("--- Dimensi Data Training ---")
print("Fitur (X_train) :", X_train.shape)
print("Target (y_train):", y_train.shape)

print("\n--- Dimensi Data Testing ---")
print("Fitur (X_test)  :", X_test.shape)
print("Target (y_test) :", y_test.shape)

# %% [markdown]
# ### Interpretasi 1
# Data wajib dipisahkan menjadi **training set** dan **testing set** agar performa model dapat diukur secara objektif pada data yang belum pernah dilihat sebelumnya. Training set dipakai untuk melatih model, sedangkan testing set dipakai untuk menguji apakah model mampu melakukan generalisasi ke data baru.
#
# Risiko terbesar jika model divalidasi menggunakan data yang sama dengan data pelatihan adalah **overestimasi performa**. Model bisa terlihat sangat bagus karena hanya menghafal pola pada data training, bukan benar-benar memahami pola umum. Akibatnya, saat diterapkan pada data nyata, performanya bisa turun karena model tidak siap menghadapi data baru.

# %%
# Import fungsi metrik evaluasi dari Scikit-Learn

# Tahap 2.1: Load dataset prediksi model
df_preds = pd.read_csv('dummy_model_predictions.csv')

# Cek nama kolom di dataset prediksi
print(df_preds.columns)

y_true = df_preds['actual_sentiment']
y_pred_alpha = df_preds['pred_alpha']

# Tahap 2.2: Hitung dan tampilkan Confusion Matrix untuk Model Alpha
cm_alpha = confusion_matrix(y_true, y_pred_alpha)
print("--- Confusion Matrix: Model Alpha ---")
print(cm_alpha)
# Catatan: Baris menunjukkan label Aktual, Kolom menunjukkan label Prediksi.

# Tahap 2.3: Tampilkan metrik klasifikasi untuk Model Alpha
print("\n--- Classification Report: Model Alpha ---")
print(classification_report(y_true, y_pred_alpha))

# %%
# Tambahan: cek proporsi label aktual pada data testing
actual_distribution = y_true.value_counts()
actual_percentage = (actual_distribution / len(y_true) * 100).round(2)

class_balance_df = pd.DataFrame({
    'jumlah': actual_distribution,
    'persentase (%)': actual_percentage
})

print("--- Distribusi Label Aktual pada Data Testing ---")
display(class_balance_df)

# %% [markdown]
# ### Interpretasi 2
# Jumlah ulasan aktual **tidak seimbang (imbalanced)**. Berdasarkan data testing, terdapat **1.695 ulasan positif (84,75%)** dan hanya **305 ulasan negatif (15,25%)**. Artinya, kelas positif jauh lebih dominan dibandingkan kelas negatif.
#
# Dalam kondisi seperti ini, metrik **accuracy** bisa menipu. Misalnya, model yang selalu menebak semua ulasan sebagai positif tetap dapat menghasilkan akurasi tinggi karena sebagian besar data memang positif. Padahal, model seperti itu gagal menjalankan fungsi bisnis utama, yaitu mendeteksi komplain atau ulasan negatif. Karena itu, evaluasi harus melihat **precision, recall, dan F1-score**, terutama untuk kelas negatif.

# %% [markdown]
# ### Interpretasi 3
# Berdasarkan **Confusion Matrix Model Alpha** `[[31, 274], [162, 1533]]`, terdapat **274 kasus kebocoran komplain**, yaitu ulasan yang sebenarnya **negatif** tetapi diprediksi **positif**. Ini adalah kondisi yang paling berbahaya bagi bisnis karena komplain pelanggan dapat terlewat dan tidak ditangani oleh tim Customer Service.
#
# Metrik yang paling perlu dioptimalkan untuk mencegah kelalaian ini adalah **recall pada kelas negatif**. Recall negatif menunjukkan seberapa banyak komplain yang berhasil tertangkap dari seluruh komplain yang benar-benar ada. Pada Model Alpha, recall kelas negatif hanya **0,10** atau sekitar **10%**, sehingga kemampuan model ini dalam menangkap komplain masih sangat lemah.

# %%
y_pred_beta = df_preds['pred_beta']
y_pred_gamma = df_preds['pred_gamma']

# Tahap 3.1: Tampilkan classification_report untuk Model Beta
print("--- Classification Report: Model Beta ---")
print(classification_report(y_true, y_pred_beta))

# Tahap 3.2: Tampilkan classification_report untuk Model Gamma
print("\n--- Classification Report: Model Gamma ---")
print(classification_report(y_true, y_pred_gamma))

# %%
# Tambahan: ringkasan perbandingan akurasi training vs testing dan recall negatif

summary_df = pd.DataFrame({
    'model': ['Alpha', 'Beta', 'Gamma'],
    'training_accuracy': [0.99, 0.55, 0.92],
    'testing_accuracy': [
        round(accuracy_score(y_true, y_pred_alpha), 2),
        round(accuracy_score(y_true, y_pred_beta), 2),
        round(accuracy_score(y_true, y_pred_gamma), 2)
    ],
    'negative_recall': [0.10, 0.46, 0.83]
})

summary_df['gap_train_test'] = (
    summary_df['training_accuracy'] - summary_df['testing_accuracy']).round(2)
summary_df

# %% [markdown]
# ### Interpretasi 4
# Jika dibandingkan antara **training accuracy** dan **testing accuracy**, maka diagnosis kesehatan model adalah sebagai berikut.
#
# **Model Alpha** mengalami **overfitting**. Training accuracy-nya sangat tinggi, yaitu **99%**, tetapi testing accuracy hanya sekitar **78%**. Selisih yang besar ini menunjukkan model terlalu menyesuaikan diri dengan data training dan tidak cukup baik dalam melakukan generalisasi pada data baru. Selain itu, recall kelas negatifnya juga sangat rendah.
#
# **Model Beta** mengalami **underfitting**. Training accuracy **55%** dan testing accuracy sekitar **54%**, yang sama-sama rendah. Ini menandakan model belum berhasil mempelajari pola penting dari data, baik pada training maupun testing.
#
# **Model Gamma** merupakan **good fit / ideal fit**. Training accuracy **92%** dan testing accuracy sekitar **91%**, dengan selisih yang sangat kecil. Hal ini menunjukkan model cukup baik mempelajari pola pada data training dan tetap mampu mempertahankan performa pada data testing. Selain itu, recall kelas negatif Model Gamma juga paling tinggi, yaitu **0,83**.

# %% [markdown]
# ### Interpretasi 5: Kesimpulan Akhir
# Model yang **paling layak di-deploy ke tahap produksi adalah Model Gamma**. Alasannya, model ini memiliki kombinasi performa yang paling seimbang: **testing accuracy tinggi (91%)**, **precision dan F1-score yang kuat**, serta yang paling penting **recall kelas negatif tertinggi (83%)**. Artinya, Model Gamma paling mampu menangkap komplain pelanggan dibandingkan dua model lainnya.
#
# Dari sisi kesehatan model, Gamma juga termasuk **good fit** karena selisih antara training accuracy (**92%**) dan testing accuracy (**91%**) sangat kecil. Sebaliknya, Alpha cenderung overfitting dan gagal mendeteksi banyak komplain, sedangkan Beta underfitting dan performanya terlalu rendah secara umum. Dengan demikian, untuk kebutuhan bisnis yang menekankan deteksi komplain agar tidak terabaikan, **Model Gamma adalah pilihan terbaik**.

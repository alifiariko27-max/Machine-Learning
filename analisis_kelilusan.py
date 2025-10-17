import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt # Pustaka untuk menampilkan plot
from sklearn.model_selection import train_test_split

# =================================================================
# Langkah 2 — Collection
# =================================================================
print("--- Langkah 2: Membaca Dataset ---")
df = pd.read_csv("kelulusan_mahasiswa.csv")
print("Informasi Dataset:")
print(df.info())
print("\n5 Data Pertama:")
print(df.head())
print("\n" + "="*50 + "\n")


# =================================================================
# Langkah 3 — Cleaning
# =================================================================
print("--- Langkah 3: Pembersihan Data ---")
print("Jumlah Missing Values per Kolom:")
print(df.isnull().sum())

# Hapus data duplikat (jika ada)
df = df.drop_duplicates()
print("\nDataset setelah menghapus duplikat (jika ada).")

# Identifikasi outlier dengan boxplot
print("\nMenampilkan Boxplot untuk IPK...")
plt.figure(figsize=(8, 5)) # Membuat kanvas gambar baru
sns.boxplot(x=df['IPK'])
plt.title('Boxplot untuk Deteksi Outlier pada IPK')
plt.show() # <-- Perintah untuk menampilkan gambar
print("\n" + "="*50 + "\n")


# =================================================================
# Langkah 4 — Exploratory Data Analysis (EDA)
# =================================================================
print("--- Langkah 4: Exploratory Data Analysis (EDA) ---")
print("Statistik Deskriptif:")
print(df.describe())

# Buat histogram distribusi IPK
print("\nMenampilkan Histogram Distribusi IPK...")
plt.figure(figsize=(8, 5))
sns.histplot(df['IPK'], bins=10, kde=True)
plt.title('Histogram Distribusi IPK')
plt.xlabel('IPK')
plt.ylabel('Frekuensi')
plt.show() # <-- Tampilkan gambar

# Visualisasi scatterplot (IPK vs Waktu Belajar)
print("\nMenampilkan Scatterplot IPK vs Waktu Belajar...")
plt.figure(figsize=(8, 6))
sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus', palette='viridis', s=100)
plt.title('Hubungan antara IPK, Waktu Belajar, dan Kelulusan')
plt.xlabel('Indeks Prestasi Kumulatif (IPK)')
plt.ylabel('Waktu Belajar (Jam)')
plt.show() # <-- Tampilkan gambar

# Tampilkan heatmap korelasi
print("\nMenampilkan Heatmap Korelasi...")
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Heatmap Korelasi antar Fitur')
plt.show() # <-- Tampilkan gambar
print("\n" + "="*50 + "\n")


# =================================================================
# Langkah 5 — Feature Engineering
# =================================================================
print("--- Langkah 5: Feature Engineering ---")
df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14  # Asumsi 14 pertemuan
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']

# Simpan ke file CSV baru
df.to_csv("processed_kelulusan.csv", index=False)
print("Fitur baru ('Rasio_Absensi', 'IPK_x_Study') telah dibuat.")
print("Dataset yang telah diproses disimpan sebagai 'processed_kelulusan.csv'.")
print("\nData setelah Feature Engineering:")
print(df.head())
print("\n" + "="*50 + "\n")


# =================================================================
# Langkah 6 — Splitting Dataset
# =================================================================
print("--- Langkah 6: Membagi Dataset ---")
X = df.drop('Lulus', axis=1)
y = df['Lulus']

# Bagi menjadi data latih (70%) dan data sementara (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# Bagi data sementara menjadi data validasi (15%) dan data tes (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Ukuran dataset Latih (Train): {X_train.shape}")
print(f"Ukuran dataset Validasi (Validation): {X_val.shape}")
print(f"Ukuran dataset Tes (Test): {X_test.shape}")
print("\nAnalisis Selesai! ✅")
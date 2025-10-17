# ==============================================================================
# Import Pustaka yang Diperlukan
# ==============================================================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import io

# ==============================================================================
# Langkah 2 — Collection (Data dimasukkan langsung ke dalam kode)
# ==============================================================================
print("--- Langkah 2: Collection ---")
csv_data = """IPK,Jumlah_Absensi,Waktu_Belajar_Jam,Lulus
3.8,3,10,1
2.5,8,5,0
3.4,4,7,1
2.1,12,2,0
3.9,2,12,1
2.8,6,4,0
3.2,5,8,1
2.7,7,3,0
3.6,4,9,1
2.3,9,4,0
"""
df = pd.read_csv(io.StringIO(csv_data))
print("Dataset berhasil dibaca.")
df.info()
print("\n5 Baris Pertama Dataset:")
print(df.head())
print("-" * 40 + "\n")

# ==============================================================================
# Langkah 3 — Cleaning
# ==============================================================================
print("--- Langkah 3: Cleaning ---")
print("Pemeriksaan Missing Values:")
print(df.isnull().sum())

# Menghapus duplikat
df = df.drop_duplicates()
print("\nDataset telah diperiksa dan tidak ada data duplikat.")

# Membuat figure untuk Boxplot (akan ditampilkan serentak di akhir)
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['IPK'])
plt.title('Boxplot untuk Identifikasi Outlier pada IPK')
print("-" * 40 + "\n")

# ==============================================================================
# Langkah 4 — Exploratory Data Analysis (EDA)
# ==============================================================================
print("--- Langkah 4: EDA ---")
print("Statistik Deskriptif:")
print(df.describe())

# Membuat figure untuk visualisasi EDA lainnya
# Histogram Distribusi IPK
plt.figure(figsize=(8, 5))
sns.histplot(df['IPK'], bins=5, kde=True)
plt.title('Distribusi IPK Mahasiswa')

# Scatterplot IPK vs Waktu Belajar
plt.figure(figsize=(8, 6))
sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus', s=100, style='Lulus', markers=['X', 'o'])
plt.title('Hubungan IPK & Waktu Belajar vs Kelulusan')
plt.legend(title='Status', labels=['Tidak Lulus', 'Lulus'])

# Heatmap Korelasi
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Heatmap Korelasi Antar Variabel')
print("\nVisualisasi EDA telah dibuat dan akan ditampilkan di akhir.")
print("-" * 40 + "\n")

# ==============================================================================
# Langkah 5 — Feature Engineering
# ==============================================================================
print("--- Langkah 5: Feature Engineering ---")
df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']
df.to_csv("processed_kelulusan.csv", index=False)
print("File 'processed_kelulusan.csv' berhasil dibuat.")
print("5 baris pertama data yang diproses:")
print(df.head())
print("-" * 40 + "\n")

# ==============================================================================
# Langkah 6 — Splitting Dataset (Dengan Perbaikan)
# ==============================================================================
print("--- Langkah 6: Splitting Dataset ---")
X = df.drop('Lulus', axis=1)
y = df['Lulus']

# Membagi menjadi Train (70%) dan Temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# Membagi Temp menjadi Validation (15%) dan Test (15%)
# 'stratify' dihilangkan di sini untuk menghindari error
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

print("Dataset berhasil dibagi.")
print(f"Ukuran Train set:      {X_train.shape}")
print(f"Ukuran Validation set: {X_val.shape}")
print(f"Ukuran Test set:       {X_test.shape}")
print("-" * 40 + "\n")

# ==============================================================================
# Menampilkan semua plot yang telah dibuat
# ==============================================================================
print("Menampilkan semua hasil visualisasi...")
plt.tight_layout()
plt.show()
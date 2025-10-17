# Nama file: inspeksi_model.py

import joblib

# 1. Muat 'resep rahasia' dari file .pkl ke dalam sebuah variabel
print("Mencoba memuat model dari rf_model.pkl...")
model = joblib.load("rf_model.pkl")
print("Model berhasil dimuat!")

print("\n===============================================")
print("         INSPEKSI ISI MODEL         ")
print("===============================================")

# 2. Tampilkan keseluruhan struktur model (pipeline)
print("\n[ Tampilan 1: Struktur Keseluruhan Model ]")
print("Ini adalah blueprint lengkap dari model Anda:")
print(model)

# 3. Tampilkan langkah-langkah yang ada di dalam pipeline
print("\n[ Tampilan 2: Langkah-langkah di Dalam Pipeline ]")
print("Model Anda terdiri dari beberapa tahap:")
# Menggunakan .named_steps untuk melihat nama setiap tahap
for nama_langkah, komponen in model.named_steps.items():
    print(f"- {nama_langkah}")

# 4. Tampilkan detail dari salah satu langkah (misalnya, model Random Forest-nya)
print("\n[ Tampilan 3: Detail Spesifik dari 'Juru Masak' (Random Forest) ]")
print("Ini adalah setelan akhir dari model Random Forest Anda setelah di-tuning:")
# Mengakses classifier (model rf) di dalam pipeline
random_forest_classifier = model.named_steps['classifier']
print(random_forest_classifier)

print("\n===============================================")
print("            Inspeksi Selesai            ")
print("===============================================")
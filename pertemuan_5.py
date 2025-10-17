# ==============================================================================
# run_pipeline.py (Versi yang Benar)
#
# Skrip lengkap untuk melatih, tuning, evaluasi, dan menyimpan model 
# Random Forest, termasuk analisis pentingnya fitur.
# Kode ini sudah memperbaiki error ValueError dengan memodifikasi cara 
# pembagian data.
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from io import StringIO

# --- Langkah 0: Simulasi Data (agar skrip mandiri) ---
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
# Membuat DataFrame dan feature engineering awal
df_raw = pd.read_csv(StringIO(csv_data))
df = df_raw.copy()
df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']
print("--- Data Awal Siap Pakai ---")
print(df.head(), "\n")


# --- Langkah 1: Muat dan Bagi Data (INI BAGIAN YANG DIPERBAIKI) ---
print("--- Langkah 1: Membagi Dataset (Train & Test) ---")
from sklearn.model_selection import train_test_split

X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# PERBAIKAN: Data terlalu kecil untuk 3 split (train/val/test). 
# Cukup bagi menjadi 2 bagian (train/test) untuk menghindari error.
# Proses validasi akan ditangani oleh Cross-Validation (Langkah 3 & 4).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print(f"Ukuran Train Set: {X_train.shape}")
print(f"Ukuran Test Set: {X_test.shape}\n")


# --- Langkah 2: Pipeline & Baseline Random Forest ---
print("--- Langkah 2: Membangun Pipeline Preprocessing & Model ---")
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

num_cols = X_train.select_dtypes(include="number").columns

preprocessor = ColumnTransformer([
    ("numeric_transformer", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),
], remainder="drop")

# Model baseline (untuned)
rf_baseline = RandomForestClassifier(
    n_estimators=300, max_features="sqrt",
    class_weight="balanced", random_state=42
)

pipe_baseline = Pipeline([("preprocessor", preprocessor), ("classifier", rf_baseline)])
print("Pipeline berhasil dibuat.\n")


# --- Langkah 3: Validasi Silang (Cross-Validation) ---
print("--- Langkah 3: Evaluasi Baseline dengan Cross-Validation ---")
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Gunakan n_splits=3 karena data latih hanya 7
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(pipe_baseline, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
print(f"CV F1-macro (Baseline): {scores.mean():.4f} ± {scores.std():.4f}\n")


# --- Langkah 4: Tuning Ringkas (GridSearchCV) ---
print("--- Langkah 4: Mencari Hyperparameter Terbaik dengan GridSearchCV ---")
from sklearn.model_selection import GridSearchCV

param_grid = {
  "classifier__max_depth": [None, 12, 20],
  "classifier__min_samples_split": [2, 3, 5]
}

gs = GridSearchCV(pipe_baseline, param_grid=param_grid, cv=skf,
                  scoring="f1_macro", n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)

print(f"Parameter terbaik: {gs.best_params_}")
print(f"CV F1-macro (Tuned): {gs.best_score_:.4f}\n")


# --- Langkah 5: Evaluasi Akhir (Test Set) ---
print("--- Langkah 5: Evaluasi Final pada Test Set ---")
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve

final_model = gs.best_estimator_

y_test_pred = final_model.predict(X_test)
y_test_proba = final_model.predict_proba(X_test)[:, 1]

print(f"F1-Score (Test): {f1_score(y_test, y_test_pred, average='macro'):.4f}")
print("ROC-AUC Score (Test):", roc_auc_score(y_test, y_test_proba))
print("\nLaporan Klasifikasi (Test):\n", classification_report(y_test, y_test_pred, digits=3))
print("Confusion Matrix (Test):\n", confusion_matrix(y_test, y_test_pred))

# Membuat dan menyimpan plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
plt.figure(); plt.plot(fpr, tpr, label='ROC Curve'); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve (test)")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--'); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.savefig("roc_test.png", dpi=120)

# Membuat dan menyimpan plot Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
plt.figure(); plt.plot(rec, prec, label='PR Curve'); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve (test)")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig("pr_test.png", dpi=120)

print("\nPlot ROC dan PR berhasil disimpan sebagai roc_test.png dan pr_test.png.\n")
# plt.show() # Hilangkan komentar ini jika Anda ingin plot muncul di layar


# --- Langkah 6: Pentingnya Fitur (Feature Importance) ---
print("--- Langkah 6: Menganalisis Pentingnya Fitur ---")
try:
    importances = final_model.named_steps["classifier"].feature_importances_
    feature_names = final_model.named_steps["preprocessor"].get_feature_names_out()
    
    feature_importance_df = pd.DataFrame(
        {'feature': feature_names, 'importance': importances}
    ).sort_values('importance', ascending=False)

    print("Pentingnya Fitur (Gini Importance):")
    print(feature_importance_df.head(10), "\n")
except Exception as e:
    print("Feature importance tidak tersedia:", e)


# --- Langkah 7: Simpan Model ---
print("--- Langkah 7: Menyimpan Model Final ---")
joblib.dump(final_model, "rf_model.pkl")
print("Model berhasil disimpan sebagai rf_model.pkl\n")


# --- Langkah 8: Cek Inference Lokal ---
print("--- Langkah 8: Uji Coba Inference Lokal ---")
loaded_model = joblib.load("rf_model.pkl")
sample_data = pd.DataFrame([{
  "IPK": 3.4,
  "Jumlah_Absensi": 4,
  "Waktu_Belajar_Jam": 7,
  "Rasio_Absensi": 4/14,
  "IPK_x_Study": 3.4*7
}])
prediction = loaded_model.predict(sample_data)[0]
proba = loaded_model.predict_proba(sample_data)[0][1]

print(f"Prediksi untuk data sampel: {prediction} (Label: {'Lulus' if prediction==1 else 'Tidak Lulus'})")
print(f"Probabilitas Lulus: {proba:.2f}")
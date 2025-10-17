# KODE LENGKAP app.py YANG SUDAH BENAR
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Muat model saat aplikasi dimulai
MODEL = joblib.load("rf_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])
    prediction = MODEL.predict(df)[0]
    proba = MODEL.predict_proba(df)[0][1]
    
    return jsonify({
        "prediction": int(prediction),
        "label": "Lulus" if int(prediction) == 1 else "Tidak Lulus",
        "probability_lulus": float(proba)
    })

if __name__ == "__main__":
    # PERBAIKAN DITAMBAHKAN DI SINI
    app.run(port=5000, debug=True, use_reloader=False)
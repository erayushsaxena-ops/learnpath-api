from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app, origins="*", allow_headers="*", methods=["GET", "POST", "OPTIONS"])


# ── Force CORS headers on every response (handles file:// null origin) ──
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ── Handle OPTIONS preflight requests ──
@app.route("/", methods=["OPTIONS"])
@app.route("/predict", methods=["OPTIONS"])
@app.route("/predict/bulk", methods=["OPTIONS"])
def handle_options():
    response = app.make_default_options_response()
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ── Load model and encoder ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "learnpath_model.pkl"))
le    = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))


# ── Health check ──
@app.route("/")
def home():
    return jsonify({
        "status": "LearnPath ML API is running!",
        "endpoints": {
            "predict":      "POST /predict",
            "predict_bulk": "POST /predict/bulk"
        }
    })


# ── Single prediction ──
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not all(k in data for k in ["score", "attendance", "attempts"]):
            return jsonify({
                "error": "Missing fields. Need: score, attendance, attempts"
            }), 400

        sample = pd.DataFrame({
            "score":      [data["score"]],
            "attendance": [1 if data["attendance"] == "Present" else 0],
            "attempts":   [data["attempts"]]
        })

        prediction = model.predict(sample)
        result     = le.inverse_transform(prediction)

        return jsonify({
            "prediction": result[0],
            "status":     "success",
            "input":      data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── Bulk prediction ──
@app.route("/predict/bulk", methods=["POST"])
def predict_bulk():
    try:
        students = request.get_json().get("students", [])

        if len(students) > 100:
            return jsonify({"error": "Max 100 students per request"}), 400

        results = []
        for i, s in enumerate(students):
            sample = pd.DataFrame({
                "score":      [s["score"]],
                "attendance": [1 if s["attendance"] == "Present" else 0],
                "attempts":   [s.get("attempts", 2)]
            })
            pred   = le.inverse_transform(model.predict(sample))[0]
            results.append({
                "index":      i,
                "prediction": pred,
                "status":     "success"
            })

        return jsonify({
            "results": results,
            "total":   len(results),
            "status":  "success"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)


from flask import Flask, request, jsonify
import joblib, pandas as pd, os

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "learnpath_model.pkl"))
le    = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))

@app.route("/")
def home():
    return jsonify({"status": "running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data   = request.get_json()
        sample = pd.DataFrame({
            "score":      [data["score"]],
            "attendance": [1 if data["attendance"] == "Present" else 0],
            "attempts":   [data["attempts"]]
        })
        result = le.inverse_transform(model.predict(sample))
        return jsonify({"prediction": result[0], "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/health")
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

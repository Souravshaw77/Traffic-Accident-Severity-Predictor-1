import os
import sys
from flask import Flask, request, jsonify, render_template

# -------------------------------------------------------------------
# PATH FIXES: Ensure src/ is importable
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# -------------------------------------------------------------------
# MODEL DIR from environment (Render/Docker) or fallback local path
# -------------------------------------------------------------------
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_DIR = os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)
os.environ["MODEL_DIR"] = MODEL_DIR  # ensure predict_severity uses same path

# -------------------------------------------------------------------
# Import inference function
# -------------------------------------------------------------------
from src.inference.predict_single import predict_severity

# -------------------------------------------------------------------
# Flask app initialization
# -------------------------------------------------------------------
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({
                "status": "error",
                "message": "Invalid or missing JSON body"
            }), 400

        result = predict_severity(data)

        return jsonify({
            "status": "success",
            "predicted_label": result["predicted_label"],
            "probabilities": result["probabilities"],
            "risk_score": result.get("risk_score"),
            "risk_level": result.get("risk_level"),
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# -------------------------------------------------------------------
# Run server (development mode). For production, use gunicorn.
# -------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False  # Turn OFF debug for safety
    )

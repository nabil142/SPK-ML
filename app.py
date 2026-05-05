"""
ML Backend — SPK Property
Flask server on port 8000
Endpoints:
  POST /ml/train        — latih model Linear Regression
  GET  /ml/status       — cek status & metrik model
  POST /ml/predict      — prediksi skor alternatif baru
"""

from flask import Flask
from flask_cors import CORS
from routes.ml_routes import ml_bp

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})   # izinkan semua origin (dev)

app.register_blueprint(ml_bp)

if __name__ == "__main__":
    app.run(port=8000, debug=True)

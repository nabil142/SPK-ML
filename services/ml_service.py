import os
import json
import requests
import numpy as np
import joblib

from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ── CONFIG ─────────────────────────────────────
NODE_BASE_URL = os.getenv("NODE_BASE_URL", "http://localhost:5000/api/v1")
MODELS_DIR    = os.path.join(os.path.dirname(__file__), "..", "models")
META_DIR      = os.path.join(MODELS_DIR, "meta")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(META_DIR,   exist_ok=True)


# ═══════════════════════════════════════════════
# PATH GLOBAL (TANPA case_id)
# ═══════════════════════════════════════════════

def _model_path(method: str):
    return os.path.join(MODELS_DIR, f"model_global_{method}.pkl")

def _scaler_path(method: str):
    return os.path.join(MODELS_DIR, f"scaler_global_{method}.pkl")

def _meta_path(method: str):
    return os.path.join(META_DIR, f"meta_global_{method}.json")


# ═══════════════════════════════════════════════
# FETCH DATASET GLOBAL
# ═══════════════════════════════════════════════

def fetch_dataset(method: str, token: str | None = None):
    url = f"{NODE_BASE_URL}/ml/dataset?method={method}"

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    return resp.json()


# ═══════════════════════════════════════════════
# TRAIN (GLOBAL)
# ═══════════════════════════════════════════════

def train(method: str, token: str | None = None):
    raw = fetch_dataset(method, token)
    data = raw.get("data") or raw

    samples = data.get("samples", [])
    feature_names = data.get("feature_names", [])

    if len(samples) < 3:
        raise ValueError("Minimal 3 data untuk training")

    X = np.array([s["features"] for s in samples], dtype=float)
    y = np.array([float(s["score"]) for s in samples])

    if not feature_names:
        feature_names = [f"fitur_{i}" for i in range(X.shape[1])]

    # split
    if len(samples) >= 6:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)

    r2  = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))

    # SAVE GLOBAL
    joblib.dump(model,  _model_path(method))
    joblib.dump(scaler, _scaler_path(method))

    meta = {
        "method": method,
        "is_trained": True,
        "r2_score": round(r2, 6),
        "mae": round(mae, 6),
        "n_samples": len(samples),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "trained_at": datetime.now().isoformat(),
        "known_scores": sorted([float(s["score"]) for s in samples], reverse=True),
    }

    with open(_meta_path(method), "w") as f:
        json.dump(meta, f, indent=2)

    return meta


# ═══════════════════════════════════════════════
# STATUS
# ═══════════════════════════════════════════════

def get_status(method: str):
    path = _meta_path(method)

    if not os.path.exists(path):
        return {
            "is_trained": False,
            "feature_names": []
        }

    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════
# PREDICT
# ═══════════════════════════════════════════════

def predict(method: str, features: dict):
    if not os.path.exists(_model_path(method)):
        raise FileNotFoundError("Model belum dilatih")

    model  = joblib.load(_model_path(method))
    scaler = joblib.load(_scaler_path(method))

    with open(_meta_path(method)) as f:
        meta = json.load(f)

    feature_names = meta["feature_names"]

    missing = [f for f in feature_names if f not in features]
    if missing:
        raise ValueError(f"Fitur kurang: {missing}")

    X_new = np.array([[float(features[f]) for f in feature_names]])

    X_new_sc = scaler.transform(X_new)
    pred = float(model.predict(X_new_sc)[0])

    # ranking
    rank = 1
    for s in meta.get("known_scores", []):
        if pred < s:
            rank += 1

    return {
        "predicted_score": round(pred, 6),
        "estimated_rank": rank
    }
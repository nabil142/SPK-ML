from flask import Blueprint, request, jsonify
from services.ml_service import train, get_status, predict

ml_bp = Blueprint("ml", __name__)


# ── Helper token ───────────────────────────────
def _get_token():
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth.split(" ", 1)[1]
    return None


# ── Response helper ────────────────────────────
def ok(data, status=200):
    return jsonify({"status": "success", "data": data}), status

def err(message, status=400):
    return jsonify({"status": "error", "message": message}), status


# ═══════════════════════════════════════════════
# POST /ml/train
# Body: { "method": "SAW" }
# ═══════════════════════════════════════════════
@ml_bp.route("/ml/train", methods=["POST"])
def route_train():
    body = request.get_json(silent=True) or {}

    method = str(body.get("method", "SAW")).upper()

    print("=== DEBUG TRAIN ===")
    print("BODY:", body)
    print("METHOD:", method)
    print("TOKEN:", _get_token())

    try:
        result = train(
            method=method,
            token=_get_token(),
        )
        print("TRAIN RESULT:", result)

        return ok(result)

    except ValueError as e:
        print("ERROR VALUE:", str(e))
        return err(str(e), 422)

    except Exception as e:
        print("ERROR TRAIN FULL:", str(e))  # 🔥 ini penting
        return err(f"Training gagal: {str(e)}", 500)


# ═══════════════════════════════════════════════
# GET /ml/status?method=SAW
# ═══════════════════════════════════════════════
@ml_bp.route("/ml/status", methods=["GET"])
def route_status():
    method = str(request.args.get("method", "SAW")).upper()

    try:
        result = get_status(method)
        return ok(result)
    except Exception as e:
        return err(f"Gagal membaca status: {str(e)}", 500)


# ═══════════════════════════════════════════════
# POST /ml/predict
# Body:
# {
#   "method": "SAW",
#   "features": {
#     "harga_tanah": 500,
#     "fasilitas": 8,
#     "jarak_ke_fasilitas_umum": 2
#   }
# }
# ═══════════════════════════════════════════════
@ml_bp.route("/ml/predict", methods=["POST"])
def route_predict():
    body = request.get_json(silent=True) or {}

    method   = str(body.get("method", "SAW")).upper()
    features = body.get("features", {})

    if not features:
        return err("Field 'features' wajib diisi dan tidak boleh kosong.")

    try:
        result = predict(
            method=method,
            features=features,
        )
        return ok(result)

    except FileNotFoundError as e:
        return err(str(e), 404)
    except ValueError as e:
        return err(str(e), 422)
    except Exception as e:
        return err(f"Prediksi gagal: {str(e)}", 500)
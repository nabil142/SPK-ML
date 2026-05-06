from flask import Blueprint
from flask import request
from flask import jsonify

from services.ml_service import (
    train_and_predict
)

ml_bp = Blueprint(
    'ml',
    __name__,
    url_prefix='/ml'
)

# ═══════════════════════════════════════════════
# PREDICT DYNAMIC
# ═══════════════════════════════════════════════
@ml_bp.route(
    '/predict-dynamic',
    methods=['POST']
)
def predict_dynamic():

    try:

        payload = request.json

        dataset = payload.get('dataset')

        features = payload.get('features')

        result = train_and_predict(
            dataset,
            features
        )

        return jsonify(result)

    except Exception as e:

        return jsonify({
            'error': str(e)
        }), 400
from flask import Flask
from flask_cors import CORS
from routes.ml_routes import ml_bp

app = Flask(__name__)

CORS(app, resources={
    r"/*": {"origins": "*"}
})

app.register_blueprint(ml_bp)

app.post('/predict-dynamic')
def predict_dynamic(payload: dict):

    method = payload.get('method', 'SAW')

    dataset = payload.get('dataset')

    features = payload.get('features')

    result = train_and_predict(
        dataset,
        features
    )

    return result

if __name__ == "__main__":
    app.run(
        port=8000,
        debug=True
    )
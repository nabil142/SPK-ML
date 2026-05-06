import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


# ═══════════════════════════════════════════════
# NORMALISASI TRAIN
# ═══════════════════════════════════════════════
def saw_normalization_train(
    X,
    feature_types
):

    X = X.copy()

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    for i, t in enumerate(feature_types):

        if t == 'benefit':
            X[:, i] = X[:, i] / (max_vals[i] + 1e-9)

        elif t == 'cost':
            X[:, i] = min_vals[i] / (X[:, i] + 1e-9)

    return X, min_vals, max_vals


# ═══════════════════════════════════════════════
# NORMALISASI PREDICT
# ═══════════════════════════════════════════════
def saw_normalization_predict(
    X,
    feature_types,
    min_vals,
    max_vals
):

    X = X.copy()

    for i, t in enumerate(feature_types):

        if t == 'benefit':
            X[:, i] = X[:, i] / (max_vals[i] + 1e-9)

        elif t == 'cost':
            X[:, i] = min_vals[i] / (X[:, i] + 1e-9)

    return X


# ═══════════════════════════════════════════════
# TRAIN + PREDICT DYNAMIC
# ═══════════════════════════════════════════════
def train_and_predict(
    dataset,
    features
):

    samples = dataset['samples']
    feature_info = dataset['feature_info']

    if len(samples) < 3:
        raise ValueError(
            'Minimal 3 data training'
        )

    feature_names = [
        f['name']
        for f in feature_info
    ]

    feature_types = [
        f['type']
        for f in feature_info
    ]

    # X y
    X = np.array([
        s['features']
        for s in samples
    ], dtype=float)

    y = np.array([
        float(s['score'])
        for s in samples
    ])

    # normalisasi
    X, min_vals, max_vals = \
        saw_normalization_train(
            X,
            feature_types
        )

    # scaling
    scaler = StandardScaler()

    X_sc = scaler.fit_transform(X)

    # model
    model = LinearRegression()
    model.fit(X_sc, y)

    # validasi feature
    missing = [
        f for f in feature_names
        if f not in features
    ]

    if missing:
        raise ValueError(
            f'Fitur kurang: {missing}'
        )

    # input baru
    X_new = np.array([[
        float(features[f])
        for f in feature_names
    ]])

    # normalisasi predict
    X_new = saw_normalization_predict(
        X_new,
        feature_types,
        min_vals,
        max_vals
    )

    # scale
    X_new_sc = scaler.transform(X_new)

    # predict
    pred = float(
        model.predict(X_new_sc)[0]
    )

    # ranking
    known_scores = sorted(
        [float(s['score']) for s in samples],
        reverse=True
    )

    rank = 1

    for s in known_scores:

        if pred < s:
            rank += 1

    # metric
    y_pred = model.predict(X_sc)

    r2 = float(
        r2_score(y, y_pred)
    )

    mae = float(
        mean_absolute_error(y, y_pred)
    )

    return {
        'predicted_score': round(pred, 6),
        'estimated_rank': rank,
        'r2_score': round(r2, 6),
        'mae': round(mae, 6),
        'n_features': len(feature_names),
        'feature_names': feature_names
    }
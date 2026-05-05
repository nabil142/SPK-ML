# ML Backend — SPK Property

Flask server (Port 8000) yang melayani 3 endpoint Machine Learning
untuk sistem SPK Property.

## Cara Menjalankan

```bash
# 1. Buat virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 2. Install dependensi
pip install -r requirements.txt

# 3. (Opsional) Salin dan sesuaikan konfigurasi
cp .env.example .env

# 4. Jalankan server
python app.py
# Server berjalan di http://localhost:8000
```

> Pastikan Node.js backend sudah berjalan di `http://localhost:5000` sebelum
> menekan tombol **Latih Model** di frontend.

---

## Endpoints

### `POST /ml/train`
Latih model Linear Regression untuk sebuah case & metode SPK.
Python akan otomatis menarik data dari Node.js endpoint `/api/v1/ml/dataset/{case_id}?method=`.

**Request body:**
```json
{ "case_id": 1, "method": "SAW" }
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "r2_score": 0.9512,
    "mae": 0.0083,
    "n_samples": 10,
    "n_features": 3,
    "feature_names": ["Harga", "Jarak", "Fasilitas"],
    "trained_at": "2026-05-05T10:00:00"
  }
}
```

---

### `GET /ml/status?case_id=1&method=SAW`
Cek apakah model sudah dilatih dan lihat metriknya.

**Response (model ada):**
```json
{
  "status": "success",
  "data": {
    "is_trained": true,
    "r2_score": 0.9512,
    "mae": 0.0083,
    "n_samples": 10,
    "n_features": 3,
    "feature_names": ["Harga", "Jarak", "Fasilitas"],
    "trained_at": "2026-05-05T10:00:00"
  }
}
```

**Response (model belum ada):**
```json
{
  "status": "success",
  "data": { "is_trained": false, "feature_names": [], ... }
}
```

---

### `POST /ml/predict`
Prediksi skor alternatif baru berdasarkan nilai kriterianya.

**Request body:**
```json
{
  "case_id": 1,
  "method": "SAW",
  "features": {
    "Harga": 450000000,
    "Jarak": 3,
    "Fasilitas": 8
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "predicted_score": 0.8120,
    "estimated_rank": 2,
    "total_existing": 5
  }
}
```

---

## Struktur File

```
ml-backend/
├── app.py                  ← entry point Flask
├── requirements.txt
├── .env.example
├── routes/
│   └── ml_routes.py        ← 3 endpoint (train, status, predict)
├── services/
│   └── ml_service.py       ← logika ML (fetch, train, predict)
└── models/                 ← model .pkl tersimpan di sini (auto-dibuat)
    └── meta/               ← metadata JSON per model
```

## Alur Data

```
Frontend (React)
     │
     │  POST /ml/train  {case_id, method}
     ▼
Python Flask (Port 8000)
     │
     │  GET /api/v1/ml/dataset/{case_id}?method=SAW
     ▼
Node.js (Port 5000)
     │  kembalikan { feature_names, samples: [{features, score}] }
     ▼
Python → latih LinearRegression → simpan model .pkl
     │  kembalikan { r2_score, mae, ... }
     ▼
Frontend menampilkan metrik akurasi
```

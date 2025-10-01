# 📋 Dokumentasi API Klasifikasi Aduan

API untuk mengklasifikasikan teks aduan masyarakat ke dalam 4 kategori: DARURAT, PRIORITAS, UMUM, dan LAINNYA menggunakan model IndoBERT.

**Base URL:** `https://api-klasifikasi-aduan.up.railway.app`

---

## 📑 Daftar Isi

- [Informasi Umum](#-informasi-umum)
- [Kategori Label](#-kategori-label)
- [Endpoint API](#-endpoint-api)
  - [1. Root](#1-root)
  - [2. Health Check](#2-health-check)
  - [3. Statistik](#3-statistik)
  - [4. Daftar Label](#4-daftar-label)
  - [5. Prediksi Tunggal](#5-prediksi-tunggal)
  - [6. Prediksi Batch](#6-prediksi-batch)
  - [7. Prediksi dari CSV](#7-prediksi-dari-csv)
  - [8. Dokumentasi](#8-dokumentasi)
- [Error Handling](#-error-handling)
- [Contoh Penggunaan](#-contoh-penggunaan)
- [Catatan Penting](#-catatan-penting)
- [Informasi Tambahan](#-informasi-tambahan)
- [Kontak](#-kontak)

---

## 🔍 Informasi Umum

API ini menggunakan model **IndoBERT** yang telah dilatih untuk mengklasifikasikan aduan masyarakat. Model dapat memproses teks dalam Bahasa Indonesia dan memberikan prediksi dengan skor confidence.

**Fitur Utama:**
- ✅ Prediksi tunggal dan batch
- ✅ Upload dan proses file CSV
- ✅ Skor confidence untuk semua kategori
- ✅ Pemrosesan cepat dan efisien
- ✅ Statistik penggunaan API

---

## 🏷️ Kategori Label

| Label | Deskripsi |
|-------|-----------|
| **DARURAT** | Memerlukan penanganan segera (kebakaran, kecelakaan, bencana) |
| **PRIORITAS** | Perlu penanganan cepat (infrastruktur rusak, kebersihan) |
| **UMUM** | Informasi/pertanyaan umum |
| **LAINNYA** | Aduan lain yang tidak termasuk kategori di atas |

---

## 🔌 Endpoint API

### 1. Root
```http
GET /
````

**Response:**

```json
{
  "message": "Aduan Classification API",
  "version": "v1",
  "docs": "/docs"
}
```

---

### 2. Health Check

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "uptime_seconds": 3600.5,
  "total_predictions": 1250,
  "version": "v1"
}
```

---

### 3. Statistik

```http
GET /stats
```

**Response:**

```json
{
  "uptime_seconds": 7200.8,
  "total_predictions": 2500,
  "total_batch_predictions": 15000,
  "total_errors": 5,
  "model_info": {
    "loaded_at": "2025-10-01T10:30:00",
    "device": "cpu",
    "source": "huggingface"
  }
}
```

---

### 4. Daftar Label

```http
GET /labels
```

**Response:**

```json
{
  "labels": ["DARURAT", "PRIORITAS", "UMUM", "LAINNYA"],
  "descriptions": {
    "DARURAT": "Memerlukan penanganan segera (kebakaran, kecelakaan, bencana)",
    "PRIORITAS": "Perlu penanganan cepat (infrastruktur rusak, kebersihan)",
    "UMUM": "Informasi/pertanyaan umum",
    "LAINNYA": "Aduan lain yang tidak termasuk kategori di atas"
  }
}
```

---

### 5. Prediksi Tunggal

```http
POST /predict
Content-Type: application/json
```

**Body:**

```json
{
  "text": "Ada kebakaran besar di jalan sudirman, tolong kirim pemadam kebakaran segera!",
  "return_all_scores": true
}
```

**Response:**

```json
{
  "label": "DARURAT",
  "confidence": 0.9823,
  "processing_time_ms": 45.2,
  "all_scores": {
    "DARURAT": 0.9823,
    "PRIORITAS": 0.0145,
    "UMUM": 0.0021,
    "LAINNYA": 0.0011
  }
}
```

---

### 6. Prediksi Batch

```http
POST /predict/batch
Content-Type: application/json
```

**Body:**

```json
{
  "texts": [
    "Jalan rusak parah di depan kantor",
    "Ada kecelakaan mobil di tol",
    "Bagaimana cara membuat KTP?"
  ],
  "return_all_scores": true
}
```

**Response (ringkas):**

```json
{
  "predictions": [
    {"label": "PRIORITAS", "confidence": 0.8956},
    {"label": "DARURAT", "confidence": 0.9512},
    {"label": "UMUM", "confidence": 0.9234}
  ],
  "total_processed": 3,
  "total_time_ms": 45.8,
  "average_time_ms": 15.3
}
```

---

### 7. Prediksi dari CSV

```http
POST /predict/csv
Content-Type: multipart/form-data
```

**Form Data:**

* `file` (required): File CSV
* `text_column` (optional): Kolom teks (default: `teks_aduan`)

**Contoh Input CSV:**

```csv
id,teks_aduan,lokasi
1,Ada kebakaran di pasar,Jl. Sudirman
2,Jalan berlubang besar,Jl. Gatot Subroto
3,Kapan jadwal vaksinasi?,Puskesmas A
```

**Response:** File CSV hasil prediksi dengan tambahan kolom:

* `predicted_label`
* `confidence`
* `prob_darurat`
* `prob_prioritas`
* `prob_umum`
* `prob_lainnya`

---

### 8. Dokumentasi

```http
GET /docs
```

---

## ⚠️ Error Handling

* **400 Bad Request**

```json
{"error": "Missing 'text' field"}
```

* **404 Not Found**

```json
{"error": "Endpoint not found"}
```

* **500 Internal Server Error**

```json
{"error": "Internal server error"}
```

* **503 Service Unavailable**

```json
{"error": "Model not loaded"}
```

---

## 💻 Contoh Penggunaan

### Python

```python
import requests

BASE_URL = "https://api-klasifikasi-aduan.up.railway.app"

res = requests.post(f"{BASE_URL}/predict", json={"text": "Ada kebakaran di pasar"})
print(res.json())
```

### JavaScript

```javascript
fetch("https://api-klasifikasi-aduan.up.railway.app/predict", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({text: "Ada kebakaran di pasar"})
})
.then(r => r.json())
.then(console.log);
```

### cURL

```bash
curl -X POST https://api-klasifikasi-aduan.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Ada kebakaran di pasar"}'
```

---

## 📝 Catatan Penting

1. Maksimal **1000 karakter** per teks
2. Maksimal **100 teks** per batch
3. File CSV harus valid dengan encoding **UTF-8**
4. Jangan spam request (rate limiting berlaku)

---

## 🔗 Informasi Tambahan

* **Model:** IndoBERT (indobenchmark/indobert-base-p1)
* **Framework:** Flask + PyTorch + Transformers
* **Deployment:** Railway
* **Bahasa:** Indonesia

---

## 📧 Kontak

Jika ada pertanyaan atau kendala, silakan hubungi tim pengembang.

---

**© 2025 API Klasifikasi Aduan**


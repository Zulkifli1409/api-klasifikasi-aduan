# 📋 API Klasifikasi Aduan v5

API untuk mengklasifikasikan teks aduan masyarakat ke dalam **5 kategori** menggunakan model IndoBERT yang telah di-fine-tune.

**Base URL:** `https://api-klasifikasi-aduan.up.railway.app`

---

## 📑 Daftar Isi

- [Informasi Umum](#-informasi-umum)
- [Kategori Label](#️-kategori-label)
- [Mode Prediksi](#-mode-prediksi)
- [Endpoint API](#-endpoint-api)
- [Error Handling](#️-error-handling)
- [Contoh Penggunaan](#-contoh-penggunaan)
- [Catatan Penting](#-catatan-penting)
- [Deployment](#-deployment)

---

## 🔍 Informasi Umum

API ini menggunakan model **IndoBERT** (v5) yang telah dilatih untuk mengklasifikasikan aduan masyarakat dengan 5 kategori. Model dapat memproses teks dalam Bahasa Indonesia dan memberikan prediksi dengan berbagai tingkat detail analisis.

### ✨ Fitur Utama

- ✅ **3 Mode Prediksi**: Basic, Advanced, dan Expert
- ✅ Prediksi tunggal dan batch (hingga 100 teks)
- ✅ Upload dan proses file CSV
- ✅ Analisis confidence dan risk level
- ✅ Rekomendasi aksi berdasarkan kategori
- ✅ Entropy analysis untuk mengukur ketidakpastian
- ✅ Statistik penggunaan API real-time

### 🔧 Teknologi

| Komponen | Detail |
|----------|--------|
| **Model** | Zulkifli1409/aduan-model (HuggingFace) |
| **Base Model** | indobenchmark/indobert-base-p1 |
| **Framework** | Flask + PyTorch + Transformers |
| **Preprocessing** | Text cleaning, URL removal, max 150 chars |
| **Max Length** | 40 tokens |
| **Device** | CPU/CUDA auto-detect |

---

## 🏷️ Kategori Label

| Label | Deskripsi | Risk Level |
|-------|-----------|------------|
| **Pinalti** | Konten tidak pantas, kasar, atau melanggar aturan | CRITICAL |
| **Darurat** | Memerlukan penanganan segera (kebakaran, kecelakaan, bencana) | HIGH |
| **Prioritas** | Perlu penanganan cepat (infrastruktur rusak, fasilitas umum) | MEDIUM |
| **Umum** | Aduan umum yang perlu ditindaklanjuti | NORMAL |
| **Lainnya** | Informasi, pertanyaan, atau aduan lain | NORMAL |

---

## 🎯 Mode Prediksi

### 1️⃣ Basic Mode
Mode standar untuk prediksi cepat dengan informasi esensial.

**Output:**
- Label prediksi
- Confidence score
- All probabilities (opsional)
- Processing time

### 2️⃣ Advanced Mode
Mode dengan analisis tambahan untuk decision making.

**Output:**
- Primary & secondary predictions
- Confidence level (HIGH/MEDIUM/LOW)
- Risk assessment (CRITICAL/HIGH/MEDIUM/NORMAL)
- Uncertainty detection
- Review requirement flag

### 3️⃣ Expert Mode
Mode paling lengkap dengan analisis mendalam dan rekomendasi.

**Output:**
- Top 3 predictions dengan ranking
- Entropy analysis (normalized)
- Decision confidence (VERY_HIGH → VERY_LOW)
- Action recommendations per kategori
- Warning system
- Metadata lengkap

---

## 🔌 Endpoint API

### 1. Root

```http
GET /
```

**Response:**
```json
{
  "service": "Aduan Classification API",
  "version": "v5 (5 Labels)",
  "model": "Zulkifli1409/aduan-model",
  "labels": ["Pinalti", "Darurat", "Prioritas", "Umum", "Lainnya"],
  "modes": ["basic", "advanced", "expert"]
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
  "model_version": "v5"
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
    "loaded_at": "2025-10-05T10:30:00",
    "device": "cpu",
    "source": "huggingface",
    "version": "v5",
    "labels": 5
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
  "labels": ["Pinalti", "Darurat", "Prioritas", "Umum", "Lainnya"],
  "descriptions": {
    "Pinalti": "Konten tidak pantas, kasar, atau melanggar aturan",
    "Darurat": "Memerlukan penanganan segera (kebakaran, kecelakaan, bencana)",
    "Prioritas": "Perlu penanganan cepat (infrastruktur rusak, fasilitas umum)",
    "Umum": "Aduan umum yang perlu ditindaklanjuti",
    "Lainnya": "Informasi, pertanyaan, atau aduan lain"
  },
  "count": 5
}
```

---

### 5. Prediksi Tunggal - Basic

```http
POST /predict
Content-Type: application/json
```

**Body:**
```json
{
  "text": "anjing kau dasar tolol",
  "return_all_scores": true
}
```

**Response:**
```json
{
  "label": "Pinalti",
  "confidence": 0.9823,
  "processing_time_ms": 45.2,
  "all_scores": {
    "Pinalti": 0.9823,
    "Darurat": 0.0089,
    "Prioritas": 0.0056,
    "Umum": 0.0021,
    "Lainnya": 0.0011
  }
}
```

---

### 6. Prediksi Tunggal - Advanced

```http
POST /predict/advanced
Content-Type: application/json
```

**Body:**
```json
{
  "text": "Ada kebakaran besar di pasar sudirman",
  "threshold": 0.7
}
```

**Response:**
```json
{
  "text_original": "Ada kebakaran besar di pasar sudirman",
  "text_cleaned": "Ada kebakaran besar di pasar sudirman",
  "prediction": {
    "primary": {
      "label": "Darurat",
      "confidence": 0.9512,
      "description": "Memerlukan penanganan segera (kebakaran, kecelakaan, bencana)"
    },
    "secondary": {
      "label": "Prioritas",
      "confidence": 0.0345,
      "description": "Perlu penanganan cepat (infrastruktur rusak, fasilitas umum)"
    }
  },
  "analysis": {
    "confidence_level": "HIGH",
    "risk_level": "HIGH",
    "uncertain": false,
    "requires_review": false
  },
  "all_probabilities": {
    "Pinalti": 0.0012,
    "Darurat": 0.9512,
    "Prioritas": 0.0345,
    "Umum": 0.0089,
    "Lainnya": 0.0042
  },
  "metadata": {
    "processing_time_ms": 52.3,
    "threshold_used": 0.7,
    "model_version": "v5"
  }
}
```

---

### 7. Prediksi Tunggal - Expert

```http
POST /predict/expert
Content-Type: application/json
```

**Body:**
```json
{
  "text": "Jalan berlubang besar di jalan merdeka",
  "include_attention": false
}
```

**Response:**
```json
{
  "input": {
    "text_original": "Jalan berlubang besar di jalan merdeka",
    "text_cleaned": "Jalan berlubang besar di jalan merdeka",
    "text_length": 39
  },
  "classification": {
    "primary_prediction": {
      "rank": 1,
      "label": "Prioritas",
      "confidence": 0.8734,
      "description": "Perlu penanganan cepat (infrastruktur rusak, fasilitas umum)"
    },
    "alternative_predictions": [
      {
        "rank": 2,
        "label": "Umum",
        "confidence": 0.0956,
        "description": "Aduan umum yang perlu ditindaklanjuti"
      },
      {
        "rank": 3,
        "label": "Darurat",
        "confidence": 0.0189,
        "description": "Memerlukan penanganan segera (kebakaran, kecelakaan, bencana)"
      }
    ],
    "decision_confidence": "HIGH"
  },
  "probability_analysis": {
    "all_probabilities": {
      "Pinalti": {"value": 0.0023, "percentage": "0.23%"},
      "Darurat": {"value": 0.0189, "percentage": "1.89%"},
      "Prioritas": {"value": 0.8734, "percentage": "87.34%"},
      "Umum": {"value": 0.0956, "percentage": "9.56%"},
      "Lainnya": {"value": 0.0098, "percentage": "0.98%"}
    },
    "entropy": {
      "value": 0.4523,
      "normalized": 0.2809,
      "interpretation": "Low"
    }
  },
  "recommendations": {
    "actions": [
      "Jadwalkan penanganan dalam 24 jam",
      "Alokasikan sumber daya",
      "Monitor perkembangan"
    ],
    "warnings": ["No warnings"],
    "review_required": false
  },
  "metadata": {
    "processing_time_ms": 58.7,
    "model_version": "v5",
    "model_type": "expert",
    "timestamp": "2025-10-05T14:23:45.123456"
  }
}
```

---

### 8. Prediksi Batch

```http
POST /predict/batch
Content-Type: application/json
```

**Body:**
```json
{
  "texts": [
    "anjing kau dasar bodoh",
    "Ada kebakaran di pasar",
    "Jalan berlubang besar",
    "Kapan jadwal vaksinasi?"
  ],
  "mode": "advanced"
}
```

**Catatan:** Mode yang tersedia: `basic`, `advanced`, `expert`

**Response:**
```json
{
  "predictions": [
    {
      "prediction": {
        "primary": {"label": "Pinalti", "confidence": 0.9823}
      }
    },
    {
      "prediction": {
        "primary": {"label": "Darurat", "confidence": 0.9512}
      }
    },
    {
      "prediction": {
        "primary": {"label": "Prioritas", "confidence": 0.8734}
      }
    },
    {
      "prediction": {
        "primary": {"label": "Lainnya", "confidence": 0.9156}
      }
    }
  ],
  "total_processed": 4,
  "mode": "advanced",
  "total_time_ms": 189.4,
  "average_time_ms": 47.35
}
```

---

### 9. Prediksi dari CSV

```http
POST /predict/csv
Content-Type: multipart/form-data
```

**Form Data:**
- `file` (required): File CSV
- `text_column` (optional): Nama kolom teks (default: `teks_aduan`)
- `mode` (optional): Mode prediksi - `basic`, `advanced`, atau `expert` (default: `basic`)

**Contoh Input CSV:**

```csv
id,teks_aduan,lokasi
1,anjing kau dasar tolol,Online
2,Ada kebakaran di pasar,Jl. Sudirman
3,Jalan berlubang besar,Jl. Gatot Subroto
4,Kapan jadwal vaksinasi?,Puskesmas A
```

**Output (mode=basic):**
```csv
id,teks_aduan,lokasi,predicted_label,confidence
1,anjing kau dasar tolol,Online,Pinalti,0.9823
2,Ada kebakaran di pasar,Jl. Sudirman,Darurat,0.9512
3,Jalan berlubang besar,Jl. Gatot Subroto,Prioritas,0.8734
4,Kapan jadwal vaksinasi?,Puskesmas A,Lainnya,0.9156
```

**Output (mode=advanced):**
```csv
id,teks_aduan,lokasi,predicted_label,confidence,confidence_level,risk_level
1,anjing kau dasar tolol,Online,Pinalti,0.9823,HIGH,CRITICAL
2,Ada kebakaran di pasar,Jl. Sudirman,Darurat,0.9512,HIGH,HIGH
3,Jalan berlubang besar,Jl. Gatot Subroto,Prioritas,0.8734,HIGH,MEDIUM
4,Kapan jadwal vaksinasi?,Puskesmas A,Lainnya,0.9156,HIGH,NORMAL
```

**Output (mode=expert):**
```csv
id,teks_aduan,lokasi,predicted_label,confidence,decision_confidence,entropy
1,anjing kau dasar tolol,Online,Pinalti,0.9823,VERY_HIGH,0.0234
2,Ada kebakaran di pasar,Jl. Sudirman,Darurat,0.9512,VERY_HIGH,0.0567
3,Jalan berlubang besar,Jl. Gatot Subroto,Prioritas,0.8734,HIGH,0.2809
4,Kapan jadwal vaksinasi?,Puskesmas A,Lainnya,0.9156,VERY_HIGH,0.1234
```

**Response:** File CSV hasil prediksi akan di-download otomatis dengan nama:
- `predictions_basic_YYYYMMDD_HHMMSS.csv`
- `predictions_advanced_YYYYMMDD_HHMMSS.csv`
- `predictions_expert_YYYYMMDD_HHMMSS.csv`

---

### 10. Dokumentasi

```http
GET /docs
```

Menampilkan dokumentasi lengkap API dalam format JSON.

---

## ⚠️ Error Handling

### 400 Bad Request

```json
{
  "error": "Missing 'text' field"
}
```

```json
{
  "error": "Text cannot be empty"
}
```

```json
{
  "error": "Maximum 100 texts per batch"
}
```

```json
{
  "error": "Invalid mode. Use: basic, advanced, or expert"
}
```

### 404 Not Found

```json
{
  "error": "Endpoint not found",
  "docs": "/docs"
}
```

### 500 Internal Server Error

```json
{
  "error": "Internal server error"
}
```

### 503 Service Unavailable

```json
{
  "error": "Model not loaded"
}
```

---

## 💻 Contoh Penggunaan

### Python - Basic

```python
import requests

BASE_URL = "https://api-klasifikasi-aduan.up.railway.app"

# Prediksi basic
response = requests.post(
    f"{BASE_URL}/predict",
    json={"text": "Ada kebakaran di pasar"}
)
print(response.json())
```

### Python - Advanced

```python
# Prediksi advanced
response = requests.post(
    f"{BASE_URL}/predict/advanced",
    json={
        "text": "Ada kebakaran besar di pasar",
        "threshold": 0.7
    }
)
result = response.json()
print(f"Label: {result['prediction']['primary']['label']}")
print(f"Risk Level: {result['analysis']['risk_level']}")
```

### Python - Expert

```python
# Prediksi expert
response = requests.post(
    f"{BASE_URL}/predict/expert",
    json={"text": "Jalan berlubang besar"}
)
result = response.json()
print(f"Label: {result['classification']['primary_prediction']['label']}")
print(f"Confidence: {result['classification']['decision_confidence']}")
print(f"Rekomendasi: {result['recommendations']['actions']}")
```

### Python - Batch

```python
# Batch prediction dengan mode advanced
texts = [
    "anjing kau dasar bodoh",
    "Ada kebakaran di pasar",
    "Jalan berlubang besar",
    "Kapan jadwal vaksinasi?"
]

response = requests.post(
    f"{BASE_URL}/predict/batch",
    json={"texts": texts, "mode": "advanced"}
)
results = response.json()
print(f"Total processed: {results['total_processed']}")
for pred in results['predictions']:
    print(pred['prediction']['primary'])
```

### Python - CSV Upload

```python
# Upload CSV
with open('aduan.csv', 'rb') as f:
    response = requests.post(
        f"{BASE_URL}/predict/csv",
        files={'file': f},
        data={'text_column': 'teks_aduan', 'mode': 'expert'}
    )

# Save hasil
with open('hasil_prediksi.csv', 'wb') as f:
    f.write(response.content)
```

### JavaScript/Node.js - Basic

```javascript
const fetch = require('node-fetch');

const BASE_URL = "https://api-klasifikasi-aduan.up.railway.app";

// Prediksi basic
fetch(`${BASE_URL}/predict`, {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({text: "Ada kebakaran di pasar"})
})
.then(r => r.json())
.then(console.log);
```

### JavaScript - Advanced

```javascript
// Prediksi advanced
fetch(`${BASE_URL}/predict/advanced`, {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({
    text: "Ada kebakaran besar di pasar",
    threshold: 0.7
  })
})
.then(r => r.json())
.then(result => {
  console.log(`Label: ${result.prediction.primary.label}`);
  console.log(`Risk: ${result.analysis.risk_level}`);
});
```

### cURL - Basic

```bash
curl -X POST https://api-klasifikasi-aduan.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Ada kebakaran di pasar"}'
```

### cURL - Advanced

```bash
curl -X POST https://api-klasifikasi-aduan.up.railway.app/predict/advanced \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Ada kebakaran besar di pasar",
    "threshold": 0.7
  }'
```

### cURL - Expert

```bash
curl -X POST https://api-klasifikasi-aduan.up.railway.app/predict/expert \
  -H "Content-Type: application/json" \
  -d '{"text": "Jalan berlubang besar"}'
```

### cURL - Batch

```bash
curl -X POST https://api-klasifikasi-aduan.up.railway.app/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Ada kebakaran", "Jalan rusak"],
    "mode": "advanced"
  }'
```

### cURL - CSV Upload

```bash
curl -X POST https://api-klasifikasi-aduan.up.railway.app/predict/csv \
  -F "file=@aduan.csv" \
  -F "text_column=teks_aduan" \
  -F "mode=expert" \
  -o hasil_prediksi.csv
```

---

## 📝 Catatan Penting

### Batasan

| Item | Limit |
|------|-------|
| Maksimal karakter per teks | 1000 karakter |
| Maksimal teks per batch | 100 teks |
| File CSV encoding | UTF-8 |
| Max length token | 40 tokens |

### Best Practices

1. **Pilih mode yang sesuai:**
   - Gunakan `basic` untuk aplikasi real-time yang butuh respons cepat
   - Gunakan `advanced` untuk sistem yang memerlukan analisis risk level
   - Gunakan `expert` untuk decision support system yang memerlukan analisis lengkap

2. **Batch processing:**
   - Untuk volume besar, gunakan batch endpoint atau CSV upload
   - Batch lebih efisien dari pada multiple single requests

3. **Threshold tuning (advanced mode):**
   - Default: 0.7
   - Tingkatkan untuk mengurangi false positive
   - Turunkan untuk mengurangi false negative

4. **Error handling:**
   - Selalu cek status code
   - Implementasi retry logic untuk 500/503 errors
   - Log error untuk debugging

### Rekomendasi Penggunaan

| Use Case | Mode yang Disarankan |
|----------|---------------------|
| Chat bot real-time | Basic |
| Sistem ticketing | Advanced |
| Dashboard monitoring | Advanced |
| Audit & compliance | Expert |
| Research & analysis | Expert |

---

## 🚀 Deployment

### Requirements

```txt
flask==3.0.0
flask-cors==4.0.0
torch==2.1.0
transformers==4.35.0
safetensors==0.4.0
pandas==2.1.0
```

### Environment Variables

```bash
PORT=8000  # Default port
```

### Local Development

```bash
# Clone repository
git clone <repo-url>
cd <repo-name>

# Install dependencies
pip install -r requirements.txt

# Run server
python app.py
```

Server akan berjalan di `http://localhost:8000`

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "app.py"]
```

```bash
# Build
docker build -t aduan-api .

# Run
docker run -p 8000:8000 aduan-api
```

### Railway Deployment

1. Push code ke GitHub
2. Connect repository di Railway
3. Railway akan auto-detect Flask app
4. Set environment variable jika diperlukan
5. Deploy!

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Average latency (basic) | ~40-50ms |
| Average latency (advanced) | ~50-60ms |
| Average latency (expert) | ~55-65ms |
| Throughput | ~20-25 req/sec |
| Model size | ~400MB |

---

## 🔗 Links

- **Model:** [Zulkifli1409/aduan-model](https://huggingface.co/Zulkifli1409/aduan-model)
- **Base Model:** [indobenchmark/indobert-base-p1](https://huggingface.co/indobenchmark/indobert-base-p1)
- **API Docs:** `/docs` endpoint

---

## 📧 Support

Jika ada pertanyaan, bug report, atau feature request, silak
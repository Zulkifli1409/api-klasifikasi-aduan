# Aduan Classification API Documentation

## Overview
Advanced FastAPI implementation for Indonesian complaint text classification with production-ready features.

## Features
- ✅ Async processing for high performance
- ✅ Batch prediction with automatic batching
- ✅ CSV file upload and processing
- ✅ Model caching and lazy loading
- ✅ Health checks and monitoring
- ✅ Comprehensive error handling
- ✅ OpenAPI/Swagger documentation
- ✅ CORS support
- ✅ Request validation with Pydantic

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Model Files
Ensure you have one of the following:
- Local: `best_model_advanced.safetensors` in the same directory
- Remote: Model uploaded to Hugging Face Hub at `Zulkifli1409/aduan-model`

### 3. Run the Server

**Development:**
```bash
python api.py
```

**Production with Uvicorn:**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

**Production with Gunicorn (Linux):**
```bash
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

## API Endpoints

### Base URL
```
http://localhost:8000
```

### Interactive Documentation
- Swagger UI: `http://localhost:8000/v1/docs`
- ReDoc: `http://localhost:8000/v1/redoc`
- OpenAPI JSON: `http://localhost:8000/v1/openapi.json`

---

## Endpoints Reference

### 1. Health Check
**GET** `/v1/health`

Check API health status and model information.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "uptime_seconds": 1234.56,
  "total_predictions": 100,
  "version": "v1"
}
```

---

### 2. Single Prediction
**POST** `/v1/predict`

Classify a single text.

**Request Body:**
```json
{
  "text": "Ada kebakaran besar di pasar tolong cepat",
  "return_all_scores": true
}
```

**Response:**
```json
{
  "label": "DARURAT",
  "confidence": 0.7429,
  "all_scores": {
    "DARURAT": 0.7429,
    "PRIORITAS": 0.1648,
    "UMUM": 0.0468,
    "LAINNYA": 0.0456
  },
  "processing_time_ms": 15.23
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Ada kebakaran besar di pasar tolong cepat",
    "return_all_scores": true
  }'
```

---

### 3. Batch Prediction
**POST** `/v1/predict/batch`

Classify multiple texts in one request (max 100).

**Request Body:**
```json
{
  "texts": [
    "Ada kebakaran di rumah warga",
    "Jalan berlubang perlu diperbaiki",
    "Mohon info jadwal posyandu"
  ],
  "return_all_scores": true
}
```

**Response:**
```json
{
  "predictions": [
    {
      "label": "DARURAT",
      "confidence": 0.7429,
      "all_scores": {...},
      "processing_time_ms": 5.12
    },
    {
      "label": "PRIORITAS",
      "confidence": 0.7847,
      "all_scores": {...},
      "processing_time_ms": 5.12
    },
    {
      "label": "UMUM",
      "confidence": 0.7209,
      "all_scores": {...},
      "processing_time_ms": 5.12
    }
  ],
  "total_processed": 3,
  "total_time_ms": 15.36,
  "average_time_ms": 5.12
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Ada kebakaran di rumah warga",
      "Jalan berlubang perlu diperbaiki"
    ],
    "return_all_scores": true
  }'
```

---

### 4. CSV File Prediction
**POST** `/v1/predict/csv`

Upload CSV file and get predictions.

**Parameters:**
- `file`: CSV file (multipart/form-data)
- `text_column`: Column name containing text (default: "teks_aduan")

**Request (multipart/form-data):**
```bash
curl -X POST "http://localhost:8000/v1/predict/csv?text_column=teks_aduan" \
  -F "file=@data.csv"
```

**Response:**
CSV file download with added columns:
- `predicted_label`: Predicted category
- `confidence`: Confidence score
- `prob_darurat`: Probability for DARURAT
- `prob_prioritas`: Probability for PRIORITAS
- `prob_umum`: Probability for UMUM
- `prob_lainnya`: Probability for LAINNYA

---

### 5. Get Labels
**GET** `/v1/labels`

Get available classification labels and descriptions.

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

### 6. Statistics
**GET** `/v1/stats`

Get API usage statistics.

**Response:**
```json
{
  "uptime_seconds": 3600.5,
  "total_predictions": 150,
  "total_batch_predictions": 450,
  "total_errors": 2,
  "model_info": {
    "loaded_at": "2024-10-01T10:30:00",
    "device": "cuda",
    "source": "local"
  }
}
```

---

## Python Client Examples

### Example 1: Single Prediction
```python
import requests

url = "http://localhost:8000/v1/predict"
payload = {
    "text": "Ada kebakaran besar di pasar",
    "return_all_scores": True
}

response = requests.post(url, json=payload)
result = response.json()

print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Processing time: {result['processing_time_ms']:.2f}ms")
```

### Example 2: Batch Prediction
```python
import requests

url = "http://localhost:8000/v1/predict/batch"
payload = {
    "texts": [
        "Ada kebakaran di rumah warga",
        "Jalan berlubang perlu diperbaiki",
        "Mohon info jadwal posyandu"
    ],
    "return_all_scores": True
}

response = requests.post(url, json=payload)
result = response.json()

print(f"Total processed: {result['total_processed']}")
print(f"Average time: {result['average_time_ms']:.2f}ms")

for i, pred in enumerate(result['predictions']):
    print(f"\nText {i+1}: {pred['label']} ({pred['confidence']:.2%})")
```

### Example 3: CSV File Upload
```python
import requests

url = "http://localhost:8000/v1/predict/csv"
files = {'file': open('data.csv', 'rb')}
params = {'text_column': 'teks_aduan'}

response = requests.post(url, files=files, params=params)

# Save result
with open('predictions_output.csv', 'wb') as f:
    f.write(response.content)

print("Predictions saved to predictions_output.csv")
```

### Example 4: Async Client with httpx
```python
import httpx
import asyncio

async def predict_async(texts):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/v1/predict/batch",
            json={"texts": texts, "return_all_scores": True},
            timeout=30.0
        )
        return response.json()

# Usage
texts = ["Text 1", "Text 2", "Text 3"]
result = asyncio.run(predict_async(texts))
print(result)
```

---

## JavaScript/TypeScript Client Examples

### Example 1: Fetch API
```javascript
// Single prediction
async function predictSingle(text) {
    const response = await fetch('http://localhost:8000/v1/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            text: text,
            return_all_scores: true
        })
    });
    
    const result = await response.json();
    console.log('Label:', result.label);
    console.log('Confidence:', result.confidence);
    return result;
}

// Batch prediction
async function predictBatch(texts) {
    const response = await fetch('http://localhost:8000/v1/predict/batch', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            texts: texts,
            return_all_scores: true
        })
    });
    
    const result = await response.json();
    return result.predictions;
}

// Usage
predictSingle("Ada kebakaran besar di pasar");
predictBatch(["Text 1", "Text 2", "Text 3"]);
```

### Example 2: Axios
```javascript
import axios from 'axios';

const API_BASE = 'http://localhost:8000/v1';

// Single prediction
async function predict(text) {
    try {
        const response = await axios.post(`${API_BASE}/predict`, {
            text: text,
            return_all_scores: true
        });
        return response.data;
    } catch (error) {
        console.error('Prediction error:', error.response.data);
        throw error;
    }
}

// Batch prediction
async function predictBatch(texts) {
    const response = await axios.post(`${API_BASE}/predict/batch`, {
        texts: texts,
        return_all_scores: true
    });
    return response.data;
}

// CSV upload
async function predictCSV(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await axios.post(
        `${API_BASE}/predict/csv?text_column=teks_aduan`,
        formData,
        {
            headers: {
                'Content-Type': 'multipart/form-data'
            },
            responseType: 'blob'
        }
    );
    
    // Download file
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'predictions.csv');
    document.body.appendChild(link);
    link.click();
}
```

---

## React Integration Example

```typescript
import React, { useState } from 'react';
import axios from 'axios';

interface PredictionResult {
    label: string;
    confidence: number;
    all_scores?: Record<string, number>;
    processing_time_ms: number;
}

const API_BASE = 'http://localhost:8000/v1';

export const AduanClassifier: React.FC = () => {
    const [text, setText] = useState('');
    const [result, setResult] = useState<PredictionResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handlePredict = async () => {
        if (!text.trim()) {
            setError('Please enter text');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const response = await axios.post(`${API_BASE}/predict`, {
                text: text,
                return_all_scores: true
            });
            setResult(response.data);
        } catch (err: any) {
            setError(err.response?.data?.error || 'Prediction failed');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="classifier">
            <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Enter complaint text..."
                rows={5}
            />
            
            <button onClick={handlePredict} disabled={loading}>
                {loading ? 'Classifying...' : 'Classify'}
            </button>

            {error && <div className="error">{error}</div>}

            {result && (
                <div className="result">
                    <h3>Result: {result.label}</h3>
                    <p>Confidence: {(result.confidence * 100).toFixed(2)}%</p>
                    <p>Processing time: {result.processing_time_ms.toFixed(2)}ms</p>
                    
                    {result.all_scores && (
                        <div className="scores">
                            <h4>All Scores:</h4>
                            {Object.entries(result.all_scores).map(([label, score]) => (
                                <div key={label}>
                                    {label}: {(score * 100).toFixed(2)}%
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};
```

---

## Error Handling

All endpoints return structured error responses:

```json
{
  "error": "Error message",
  "detail": "Detailed error description",
  "timestamp": "2024-10-01T10:30:00"
}
```

### Common HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request (invalid input) |
| 422 | Unprocessable Entity (validation error) |
| 500 | Internal Server Error |
| 503 | Service Unavailable (model not loaded) |

---

## Performance Optimization

### 1. Batch Processing
Always use batch endpoint for multiple texts:
```python
# Good - Efficient
response = requests.post(url + "/predict/batch", json={"texts": texts})

# Bad - Inefficient
for text in texts:
    response = requests.post(url + "/predict", json={"text": text})
```

### 2. Connection Pooling
Use session for multiple requests:
```python
import requests

session = requests.Session()
for text in texts:
    response = session.post(url, json={"text": text})
```

### 3. Async Requests
Use async client for concurrent requests:
```python
import httpx
import asyncio

async def predict_concurrent(texts):
    async with httpx.AsyncClient() as client:
        tasks = [
            client.post(url, json={"text": text})
            for text in texts
        ]
        responses = await asyncio.gather(*tasks)
        return [r.json() for r in responses]
```

---

## Production Deployment

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY api.py .
COPY best_model_advanced.safetensors .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**Build and Run:**
```bash
docker build -t aduan-api .
docker run -p 8000:8000 --gpus all aduan-api
```

### Docker Compose

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - WORKERS=4
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

### Kubernetes Deployment

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aduan-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: aduan-api
  template:
    metadata:
      labels:
        app: aduan-api
    spec:
      containers:
      - name: api
        image: aduan-api:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            memory: "4Gi"
            cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: aduan-api-service
spec:
  selector:
    app: aduan-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## Monitoring and Logging

### Prometheus Metrics (Optional)

Add to requirements.txt:
```
prometheus-fastapi-instrumentator==6.1.0
```

Add to api.py:
```python
from prometheus_fastapi_instrumentator import Instrumentator

# After app creation
Instrumentator().instrument(app).expose(app)
```

Access metrics at: `http://localhost:8000/metrics`

### Logging Configuration

```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
handler = RotatingFileHandler(
    'api.log', 
    maxBytes=10000000, 
    backupCount=5
)
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))

logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

---

## Security Best Practices

1. **API Key Authentication** (for production):
```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != "your-secret-key":
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# Use in endpoints
@app.post("/v1/predict", dependencies=[Depends(verify_api_key)])
async def predict(...):
    ...
```

2. **Rate Limiting**:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/v1/predict")
@limiter.limit("10/minute")
async def predict(...):
    ...
```

3. **CORS Configuration**:
Update in production:
```python
ALLOWED_ORIGINS = [
    "https://yourdomain.com",
    "https://www.yourdomain.com"
]
```

---

## Troubleshooting

### Issue 1: Model Not Loading
```
Error: Model not loaded
```
**Solution:** Check if `best_model_advanced.safetensors` exists or HF model is accessible.

### Issue 2: CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce `BATCH_SIZE` in settings or use CPU:
```python
DEVICE = "cpu"
```

### Issue 3: Slow Predictions
**Solution:** 
- Use batch endpoint instead of single predictions
- Enable GPU if available
- Increase workers in production

### Issue 4: Connection Timeout
**Solution:** Increase timeout in client:
```python
response = requests.post(url, json=payload, timeout=60)
```

---

## Support and Contact

For issues and questions:
- Check logs: `api.log`
- Health endpoint: `/v1/health`
- Statistics endpoint: `/v1/stats`

---

## License

[Your License Here]

## Version History

- **v1.0.0** (2024-10-01): Initial release
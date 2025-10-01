from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from safetensors.torch import load_file
import time
from datetime import datetime
import logging
import pandas as pd
from io import StringIO, BytesIO
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    MODEL_NAME = "indobenchmark/indobert-base-p1"
    MODEL_PATH = "best_model_advanced.safetensors"
    HF_MODEL_NAME = "Zulkifli1409/aduan-model"
    MAX_LENGTH = 128
    BATCH_SIZE = 32
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_TEXT_LENGTH = 1000


config = Config()

# Global variables
model = None
tokenizer = None
stats = {
    "total_predictions": 0,
    "total_batch_predictions": 0,
    "total_errors": 0,
    "uptime_start": datetime.now(),
    "model_loaded_at": None,
    "model_source": None,
}

LABEL_MAP = {0: "DARURAT", 1: "PRIORITAS", 2: "UMUM", 3: "LAINNYA"}

LABEL_DESCRIPTIONS = {
    "DARURAT": "Memerlukan penanganan segera (kebakaran, kecelakaan, bencana)",
    "PRIORITAS": "Perlu penanganan cepat (infrastruktur rusak, kebersihan)",
    "UMUM": "Informasi/pertanyaan umum",
    "LAINNYA": "Aduan lain yang tidak termasuk kategori di atas",
}


# ============================================================================
# MODEL LOADING
# ============================================================================
def load_model_from_local():
    """Load model from local safetensors file"""
    try:
        logger.info("Loading model from local file...")

        tok = AutoTokenizer.from_pretrained(config.MODEL_NAME)

        cfg = AutoConfig.from_pretrained(config.MODEL_NAME)
        cfg.num_labels = 4
        cfg.hidden_dropout_prob = 0.1
        cfg.attention_probs_dropout_prob = 0.1
        cfg.classifier_dropout = 0.1

        mdl = AutoModelForSequenceClassification.from_pretrained(
            config.MODEL_NAME, config=cfg, ignore_mismatched_sizes=True
        )

        # Load trained weights
        state_dict = load_file(config.MODEL_PATH, device=str(config.DEVICE))
        mdl.load_state_dict(state_dict)

        mdl = mdl.to(config.DEVICE)
        mdl.eval()

        logger.info(f"✅ Model loaded from local on {config.DEVICE}")
        return mdl, tok, "local"

    except Exception as e:
        logger.error(f"❌ Error loading local model: {e}")
        raise


def load_model_from_hf():
    """Load model from Hugging Face Hub"""
    try:
        logger.info("Loading model from Hugging Face...")

        tok = AutoTokenizer.from_pretrained(config.HF_MODEL_NAME)

        cfg = AutoConfig.from_pretrained(config.HF_MODEL_NAME)
        cfg.num_labels = 4

        mdl = AutoModelForSequenceClassification.from_pretrained(
            config.HF_MODEL_NAME, config=cfg, ignore_mismatched_sizes=True
        )

        mdl = mdl.to(config.DEVICE)
        mdl.eval()

        logger.info(f"✅ Model loaded from HF on {config.DEVICE}")
        return mdl, tok, "huggingface"

    except Exception as e:
        logger.error(f"❌ Error loading HF model: {e}")
        raise


def initialize_model():
    """Initialize model on startup"""
    global model, tokenizer, stats

    try:
        # Try local first, fallback to HF
        try:
            model, tokenizer, source = load_model_from_local()
        except:
            model, tokenizer, source = load_model_from_hf()

        stats["model_loaded_at"] = datetime.now()
        stats["model_source"] = source

        logger.info(f"✅ Model initialized from {source}")

    except Exception as e:
        logger.error(f"❌ Failed to initialize model: {e}")
        raise


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================
def predict_single(text, return_all_scores=True):
    """Predict single text"""
    start_time = time.time()

    try:
        # Tokenize
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=config.MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(config.DEVICE)
        attention_mask = encoding["attention_mask"].to(config.DEVICE)

        # Predict
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)[0]

            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()

        result = {
            "label": LABEL_MAP[pred_idx],
            "confidence": confidence,
            "processing_time_ms": (time.time() - start_time) * 1000,
        }

        if return_all_scores:
            result["all_scores"] = {
                LABEL_MAP[i]: probs[i].item() for i in range(len(LABEL_MAP))
            }

        stats["total_predictions"] += 1
        return result

    except Exception as e:
        stats["total_errors"] += 1
        logger.error(f"Prediction error: {e}")
        raise


def predict_batch(texts, return_all_scores=True):
    """Predict multiple texts in batch"""
    results = []

    # Process in batches
    for i in range(0, len(texts), config.BATCH_SIZE):
        batch_texts = texts[i : i + config.BATCH_SIZE]

        # Tokenize batch
        encodings = tokenizer.batch_encode_plus(
            batch_texts,
            add_special_tokens=True,
            max_length=config.MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(config.DEVICE)
        attention_mask = encodings["attention_mask"].to(config.DEVICE)

        # Predict batch
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)

            pred_indices = torch.argmax(probs, dim=1).cpu().numpy()
            confidences = torch.max(probs, dim=1).values.cpu().numpy()

        # Build results
        for idx, (pred_idx, conf) in enumerate(zip(pred_indices, confidences)):
            result = {
                "label": LABEL_MAP[pred_idx],
                "confidence": float(conf),
            }

            if return_all_scores:
                result["all_scores"] = {
                    LABEL_MAP[j]: float(probs[idx][j].item())
                    for j in range(len(LABEL_MAP))
                }

            results.append(result)

    stats["total_batch_predictions"] += len(texts)
    return results


# ============================================================================
# FLASK APP
# ============================================================================
app = Flask(__name__)
CORS(app)

# Initialize model before first request
with app.app_context():
    initialize_model()


# ============================================================================
# ENDPOINTS
# ============================================================================
@app.route("/", methods=["GET"])
def root():
    """Root endpoint"""
    return jsonify(
        {"message": "Aduan Classification API", "version": "v1", "docs": "/docs"}
    )


@app.route("/health", methods=["GET"])
def health():
    """Health check"""
    uptime = (datetime.now() - stats["uptime_start"]).total_seconds()

    return jsonify(
        {
            "status": "healthy" if model is not None else "unhealthy",
            "model_loaded": model is not None,
            "device": str(config.DEVICE),
            "uptime_seconds": uptime,
            "total_predictions": stats["total_predictions"],
            "version": "v1",
        }
    )


@app.route("/stats", methods=["GET"])
def get_stats():
    """Get statistics"""
    uptime = (datetime.now() - stats["uptime_start"]).total_seconds()

    return jsonify(
        {
            "uptime_seconds": uptime,
            "total_predictions": stats["total_predictions"],
            "total_batch_predictions": stats["total_batch_predictions"],
            "total_errors": stats["total_errors"],
            "model_info": {
                "loaded_at": (
                    stats["model_loaded_at"].isoformat()
                    if stats["model_loaded_at"]
                    else None
                ),
                "device": str(config.DEVICE),
                "source": stats["model_source"],
            },
        }
    )


@app.route("/labels", methods=["GET"])
def get_labels():
    """Get available labels"""
    return jsonify(
        {"labels": list(LABEL_MAP.values()), "descriptions": LABEL_DESCRIPTIONS}
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict single text

    Body JSON:
    {
        "text": "teks aduan",
        "return_all_scores": true
    }
    """
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 503

        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        text = data["text"].strip()

        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400

        if len(text) > config.MAX_TEXT_LENGTH:
            return (
                jsonify(
                    {"error": f"Text too long (max {config.MAX_TEXT_LENGTH} chars)"}
                ),
                400,
            )

        return_all_scores = data.get("return_all_scores", True)

        result = predict_single(text, return_all_scores)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch_endpoint():
    """
    Predict multiple texts

    Body JSON:
    {
        "texts": ["teks1", "teks2", ...],
        "return_all_scores": true
    }
    """
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 503

        data = request.get_json()

        if not data or "texts" not in data:
            return jsonify({"error": "Missing 'texts' field"}), 400

        texts = data["texts"]

        if not isinstance(texts, list):
            return jsonify({"error": "'texts' must be a list"}), 400

        if len(texts) == 0:
            return jsonify({"error": "Empty texts list"}), 400

        if len(texts) > 100:
            return jsonify({"error": "Maximum 100 texts per batch"}), 400

        # Clean texts
        texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]

        if not texts:
            return jsonify({"error": "No valid texts found"}), 400

        return_all_scores = data.get("return_all_scores", True)

        start_time = time.time()
        results = predict_batch(texts, return_all_scores)
        total_time = (time.time() - start_time) * 1000

        # Add processing time
        avg_time = total_time / len(results)
        for r in results:
            r["processing_time_ms"] = avg_time

        return jsonify(
            {
                "predictions": results,
                "total_processed": len(results),
                "total_time_ms": total_time,
                "average_time_ms": avg_time,
            }
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict/csv", methods=["POST"])
def predict_csv():
    """
    Predict from CSV file

    Form data:
    - file: CSV file
    - text_column: column name (default: teks_aduan)
    """
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 503

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        text_column = request.form.get("text_column", "teks_aduan")

        # Read CSV
        df = pd.read_csv(file)

        if text_column not in df.columns:
            return jsonify({"error": f"Column '{text_column}' not found"}), 400

        # Get texts
        texts = df[text_column].dropna().astype(str).tolist()

        if not texts:
            return jsonify({"error": "No valid texts found"}), 400

        # Predict
        results = predict_batch(texts, True)

        # Add results to dataframe
        df_results = df.copy()
        df_results["predicted_label"] = [r["label"] for r in results]
        df_results["confidence"] = [r["confidence"] for r in results]

        # Add probability columns
        for label in LABEL_MAP.values():
            df_results[f"prob_{label.lower()}"] = [
                r["all_scores"][label] for r in results
            ]

        # Convert to CSV
        output = BytesIO()
        df_results.to_csv(output, index=False)
        output.seek(0)

        filename = f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

        return send_file(
            output, mimetype="text/csv", as_attachment=True, download_name=filename
        )

    except pd.errors.ParserError:
        return jsonify({"error": "Invalid CSV file"}), 400
    except Exception as e:
        logger.error(f"CSV prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/docs", methods=["GET"])
def docs():
    """API Documentation"""
    return jsonify(
        {
            "api": "Aduan Classification API",
            "version": "v1",
            "endpoints": {
                "GET /": "Root endpoint",
                "GET /health": "Health check",
                "GET /stats": "API statistics",
                "GET /labels": "Get available labels",
                "POST /predict": "Predict single text",
                "POST /predict/batch": "Predict multiple texts",
                "POST /predict/csv": "Predict from CSV file",
                "GET /docs": "This documentation",
            },
            "examples": {
                "/predict": {
                    "request": {
                        "text": "Ada kebakaran di jalan sudirman",
                        "return_all_scores": True,
                    },
                    "response": {
                        "label": "DARURAT",
                        "confidence": 0.95,
                        "all_scores": {
                            "DARURAT": 0.95,
                            "PRIORITAS": 0.03,
                            "UMUM": 0.01,
                            "LAINNYA": 0.01,
                        },
                        "processing_time_ms": 45.2,
                    },
                }
            },
        }
    )


# ============================================================================
# ERROR HANDLERS
# ============================================================================
@app.errorhandler(404)
def not_found(e):
    return (
        jsonify(
            {
                "error": "Endpoint not found",
                "message": "Check /docs for available endpoints",
            }
        ),
        404,
    )


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal error: {e}")
    return jsonify({"error": "Internal server error", "message": str(e)}), 500


# ============================================================================
# RUN SERVER
# ============================================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)

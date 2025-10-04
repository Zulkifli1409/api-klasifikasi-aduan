from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch

torch.set_num_threads(1)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import time
from datetime import datetime
import logging
import pandas as pd
from io import BytesIO
import os
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ==========================================================================
# CONFIGURATION
# ==========================================================================
class Config:
    MODEL_NAME = "indobenchmark/indobert-base-p1"
    HF_MODEL_NAME = "Zulkifli1409/aduan-model"
    MODEL_FILE = "model_v5labels.safetensors"
    MAX_LENGTH = 40  # Sesuai dengan training script
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

# 5 Labels seperti di training script
LABEL_MAP = {0: "Pinalti", 1: "Darurat", 2: "Prioritas", 3: "Umum", 4: "Lainnya"}

LABEL_DESCRIPTIONS = {
    "Pinalti": "Konten tidak pantas, kasar, atau melanggar aturan",
    "Darurat": "Memerlukan penanganan segera (kebakaran, kecelakaan, bencana)",
    "Prioritas": "Perlu penanganan cepat (infrastruktur rusak, fasilitas umum)",
    "Umum": "Aduan umum yang perlu ditindaklanjuti",
    "Lainnya": "Informasi, pertanyaan, atau aduan lain yang tidak termasuk kategori di atas",
}


# ============================================================================
# TEXT PREPROCESSING
# ============================================================================
def clean_text(text):
    """Text preprocessing (sama dengan training)"""
    text = str(text).strip()
    text = ' '.join(text.split())
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    return text[:150]


# ============================================================================
# MODEL LOADING
# ============================================================================
def load_model_from_hf():
    """Load model from Hugging Face Hub"""
    try:
        logger.info(f"Loading model from Hugging Face: {config.HF_MODEL_NAME}")

        # Load tokenizer
        tok = AutoTokenizer.from_pretrained(config.HF_MODEL_NAME, use_fast=True)

        # Load config
        cfg = AutoConfig.from_pretrained(config.HF_MODEL_NAME)
        cfg.num_labels = 5  # 5 labels

        # Load model
        mdl = AutoModelForSequenceClassification.from_pretrained(
            config.HF_MODEL_NAME,
            config=cfg,
            ignore_mismatched_sizes=True
        )

        mdl = mdl.to(config.DEVICE)
        mdl.eval()

        logger.info(f"✅ Model loaded from HF on {config.DEVICE}")
        return mdl, tok, "huggingface"

    except Exception as e:
        logger.error(f"❌ Error loading HF model: {e}")
        raise


def initialize_model():
    global model, tokenizer, stats
    try:
        model, tokenizer, source = load_model_from_hf()
        stats["model_loaded_at"] = datetime.now()
        stats["model_source"] = source
        logger.info(f"✅ Model initialized from {source}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize model: {e}")
        raise


# ============================================================================
# PREDICTION FUNCTIONS - BASIC
# ============================================================================
def predict_single(text, return_all_scores=True):
    """Predict single text - Basic version"""
    start_time = time.time()

    try:
        # Clean text
        text_clean = clean_text(text)
        
        # Tokenize
        encoding = tokenizer(
            text_clean,
            add_special_tokens=True,
            max_length=config.MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(config.DEVICE)
        attention_mask = encoding["attention_mask"].to(config.DEVICE)

        # Predict
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]

            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()

        result = {
            "label": LABEL_MAP[pred_idx],
            "confidence": confidence,
            "processing_time_ms": (time.time() - start_time) * 1000,
        }

        if return_all_scores:
            result["all_scores"] = {
                LABEL_MAP[i]: float(probs[i].item()) for i in range(len(LABEL_MAP))
            }

        stats["total_predictions"] += 1
        return result

    except Exception as e:
        stats["total_errors"] += 1
        logger.error(f"Prediction error: {e}")
        raise


# ============================================================================
# PREDICTION FUNCTIONS - ADVANCED
# ============================================================================
def predict_advanced(text, threshold=0.7):
    """
    Advanced prediction with:
    - Confidence threshold
    - Secondary predictions
    - Risk assessment
    """
    start_time = time.time()

    try:
        text_clean = clean_text(text)
        
        encoding = tokenizer(
            text_clean,
            add_special_tokens=True,
            max_length=config.MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(config.DEVICE)
        attention_mask = encoding["attention_mask"].to(config.DEVICE)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]

        # Sort predictions by probability
        sorted_indices = torch.argsort(probs, descending=True)
        
        primary_idx = sorted_indices[0].item()
        primary_conf = probs[primary_idx].item()
        
        secondary_idx = sorted_indices[1].item()
        secondary_conf = probs[secondary_idx].item()

        # Confidence level
        if primary_conf >= 0.9:
            confidence_level = "HIGH"
        elif primary_conf >= threshold:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

        # Risk assessment
        risk_level = "NORMAL"
        if LABEL_MAP[primary_idx] == "Pinalti":
            risk_level = "CRITICAL"
        elif LABEL_MAP[primary_idx] == "Darurat":
            risk_level = "HIGH"
        elif LABEL_MAP[primary_idx] == "Prioritas":
            risk_level = "MEDIUM"

        # Uncertainty flag
        uncertainty = (primary_conf - secondary_conf) < 0.2

        result = {
            "text_original": text[:100],
            "text_cleaned": text_clean[:100],
            "prediction": {
                "primary": {
                    "label": LABEL_MAP[primary_idx],
                    "confidence": primary_conf,
                    "description": LABEL_DESCRIPTIONS[LABEL_MAP[primary_idx]]
                },
                "secondary": {
                    "label": LABEL_MAP[secondary_idx],
                    "confidence": secondary_conf,
                    "description": LABEL_DESCRIPTIONS[LABEL_MAP[secondary_idx]]
                }
            },
            "analysis": {
                "confidence_level": confidence_level,
                "risk_level": risk_level,
                "uncertain": uncertainty,
                "requires_review": primary_conf < threshold or uncertainty
            },
            "all_probabilities": {
                LABEL_MAP[i]: float(probs[i].item()) for i in range(len(LABEL_MAP))
            },
            "metadata": {
                "processing_time_ms": (time.time() - start_time) * 1000,
                "threshold_used": threshold,
                "model_version": "v5"
            }
        }

        stats["total_predictions"] += 1
        return result

    except Exception as e:
        stats["total_errors"] += 1
        logger.error(f"Advanced prediction error: {e}")
        raise


# ============================================================================
# PREDICTION FUNCTIONS - EXPERT
# ============================================================================
def predict_expert(text, include_attention=False):
    """
    Expert prediction with:
    - Detailed probability distribution
    - Entropy analysis
    - Attention weights (optional)
    - Recommendation system
    """
    start_time = time.time()

    try:
        text_clean = clean_text(text)
        
        encoding = tokenizer(
            text_clean,
            add_special_tokens=True,
            max_length=config.MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(config.DEVICE)
        attention_mask = encoding["attention_mask"].to(config.DEVICE)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=include_attention
            )
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]

        # Entropy calculation (measure of uncertainty)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        max_entropy = -torch.log(torch.tensor(1.0 / len(LABEL_MAP))).item()
        normalized_entropy = entropy / max_entropy

        # Get top 3 predictions
        top_k = min(3, len(LABEL_MAP))
        top_probs, top_indices = torch.topk(probs, top_k)
        
        top_predictions = [
            {
                "rank": i + 1,
                "label": LABEL_MAP[idx.item()],
                "confidence": prob.item(),
                "description": LABEL_DESCRIPTIONS[LABEL_MAP[idx.item()]]
            }
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices))
        ]

        # Decision confidence
        primary_conf = top_predictions[0]["confidence"]
        if primary_conf >= 0.95:
            decision_confidence = "VERY_HIGH"
        elif primary_conf >= 0.85:
            decision_confidence = "HIGH"
        elif primary_conf >= 0.70:
            decision_confidence = "MEDIUM"
        elif primary_conf >= 0.50:
            decision_confidence = "LOW"
        else:
            decision_confidence = "VERY_LOW"

        # Action recommendations
        recommendations = []
        primary_label = top_predictions[0]["label"]
        
        if primary_label == "Pinalti":
            recommendations.extend([
                "Flag untuk moderasi",
                "Tinjau konten secara manual",
                "Pertimbangkan tindakan pelanggaran"
            ])
        elif primary_label == "Darurat":
            recommendations.extend([
                "Prioritaskan penanganan segera",
                "Hubungi tim darurat",
                "Eskalasi ke pihak berwenang"
            ])
        elif primary_label == "Prioritas":
            recommendations.extend([
                "Jadwalkan penanganan dalam 24 jam",
                "Alokasikan sumber daya",
                "Monitor perkembangan"
            ])
        elif primary_label == "Umum":
            recommendations.extend([
                "Tambahkan ke antrian regular",
                "Tindaklanjuti sesuai SOP",
                "Update status ke pelapor"
            ])
        else:  # Lainnya
            recommendations.extend([
                "Kategorikan lebih lanjut",
                "Berikan informasi atau arahan",
                "Tutup jika tidak memerlukan tindakan"
            ])

        # Uncertainty warnings
        warnings = []
        if normalized_entropy > 0.8:
            warnings.append("High uncertainty - manual review recommended")
        if primary_conf < 0.6:
            warnings.append("Low confidence - consider human verification")
        if top_predictions[0]["confidence"] - top_predictions[1]["confidence"] < 0.15:
            warnings.append("Close predictions - ambiguous classification")

        result = {
            "input": {
                "text_original": text[:100],
                "text_cleaned": text_clean[:100],
                "text_length": len(text_clean)
            },
            "classification": {
                "primary_prediction": top_predictions[0],
                "alternative_predictions": top_predictions[1:],
                "decision_confidence": decision_confidence
            },
            "probability_analysis": {
                "all_probabilities": {
                    LABEL_MAP[i]: {
                        "value": float(probs[i].item()),
                        "percentage": f"{probs[i].item() * 100:.2f}%"
                    }
                    for i in range(len(LABEL_MAP))
                },
                "entropy": {
                    "value": entropy,
                    "normalized": normalized_entropy,
                    "interpretation": "Low" if normalized_entropy < 0.4 else "Medium" if normalized_entropy < 0.7 else "High"
                }
            },
            "recommendations": {
                "actions": recommendations,
                "warnings": warnings if warnings else ["No warnings"],
                "review_required": len(warnings) > 0 or primary_conf < 0.7
            },
            "metadata": {
                "processing_time_ms": (time.time() - start_time) * 1000,
                "model_version": "v5",
                "model_type": "expert",
                "timestamp": datetime.now().isoformat()
            }
        }

        # Add attention weights if requested
        if include_attention and outputs.attentions is not None:
            # Get last layer attention (simplified)
            last_attention = outputs.attentions[-1][0].mean(0).cpu().numpy()
            result["attention_analysis"] = {
                "available": True,
                "note": "Attention weights from last layer (averaged across heads)"
            }

        stats["total_predictions"] += 1
        return result

    except Exception as e:
        stats["total_errors"] += 1
        logger.error(f"Expert prediction error: {e}")
        raise


# ============================================================================
# BATCH PREDICTION
# ============================================================================
def predict_batch(texts, mode="basic", **kwargs):
    """Batch prediction with different modes"""
    results = []
    
    for text in texts:
        try:
            if mode == "advanced":
                result = predict_advanced(text, **kwargs)
            elif mode == "expert":
                result = predict_expert(text, **kwargs)
            else:  # basic
                result = predict_single(text, **kwargs)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            results.append({"error": str(e), "text": text[:50]})
    
    stats["total_batch_predictions"] += len(texts)
    return results


# ============================================================================
# FLASK APP
# ============================================================================
app = Flask(__name__)
CORS(app)

# Initialize model
with app.app_context():
    initialize_model()


# ============================================================================
# ENDPOINTS
# ============================================================================
@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "Aduan Classification API",
        "version": "v5 (5 Labels)",
        "model": config.HF_MODEL_NAME,
        "labels": list(LABEL_MAP.values()),
        "modes": ["basic", "advanced", "expert"],
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "labels": "/labels"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    uptime = (datetime.now() - stats["uptime_start"]).total_seconds()
    return jsonify({
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "device": str(config.DEVICE),
        "uptime_seconds": uptime,
        "total_predictions": stats["total_predictions"],
        "model_version": "v5"
    })


@app.route("/stats", methods=["GET"])
def get_stats():
    uptime = (datetime.now() - stats["uptime_start"]).total_seconds()
    return jsonify({
        "uptime_seconds": uptime,
        "total_predictions": stats["total_predictions"],
        "total_batch_predictions": stats["total_batch_predictions"],
        "total_errors": stats["total_errors"],
        "model_info": {
            "loaded_at": stats["model_loaded_at"].isoformat() if stats["model_loaded_at"] else None,
            "device": str(config.DEVICE),
            "source": stats["model_source"],
            "version": "v5",
            "labels": len(LABEL_MAP)
        }
    })


@app.route("/labels", methods=["GET"])
def get_labels():
    return jsonify({
        "labels": list(LABEL_MAP.values()),
        "descriptions": LABEL_DESCRIPTIONS,
        "count": len(LABEL_MAP)
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Basic prediction endpoint"""
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 503

        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        text = data["text"].strip()
        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400

        return_all_scores = data.get("return_all_scores", True)
        result = predict_single(text, return_all_scores)
        
        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict/advanced", methods=["POST"])
def predict_advanced_endpoint():
    """Advanced prediction with analysis"""
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 503

        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        text = data["text"].strip()
        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400

        threshold = data.get("threshold", 0.7)
        result = predict_advanced(text, threshold)
        
        return jsonify(result)

    except Exception as e:
        logger.error(f"Advanced prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict/expert", methods=["POST"])
def predict_expert_endpoint():
    """Expert prediction with detailed analysis"""
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 503

        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        text = data["text"].strip()
        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400

        include_attention = data.get("include_attention", False)
        result = predict_expert(text, include_attention)
        
        return jsonify(result)

    except Exception as e:
        logger.error(f"Expert prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch_endpoint():
    """Batch prediction with mode selection"""
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 503

        data = request.get_json()
        if not data or "texts" not in data:
            return jsonify({"error": "Missing 'texts' field"}), 400

        texts = data["texts"]
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({"error": "Invalid texts list"}), 400

        if len(texts) > 100:
            return jsonify({"error": "Maximum 100 texts per batch"}), 400

        mode = data.get("mode", "basic")  # basic, advanced, expert
        if mode not in ["basic", "advanced", "expert"]:
            return jsonify({"error": "Invalid mode. Use: basic, advanced, or expert"}), 400

        start_time = time.time()
        results = predict_batch(texts, mode=mode)
        total_time = (time.time() - start_time) * 1000

        return jsonify({
            "predictions": results,
            "total_processed": len(results),
            "mode": mode,
            "total_time_ms": total_time,
            "average_time_ms": total_time / len(results)
        })

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict/csv", methods=["POST"])
def predict_csv():
    """Predict from CSV with mode selection"""
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 503

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        text_column = request.form.get("text_column", "teks_aduan")
        mode = request.form.get("mode", "basic")

        # Read CSV
        df = pd.read_csv(file)
        if text_column not in df.columns:
            return jsonify({"error": f"Column '{text_column}' not found"}), 400

        texts = df[text_column].dropna().astype(str).tolist()
        if not texts:
            return jsonify({"error": "No valid texts found"}), 400

        # Predict
        results = predict_batch(texts, mode=mode)

        # Build result dataframe based on mode
        df_results = df.copy()
        
        if mode == "basic":
            df_results["predicted_label"] = [r.get("label", "") for r in results]
            df_results["confidence"] = [r.get("confidence", 0) for r in results]
            
        elif mode == "advanced":
            df_results["predicted_label"] = [r.get("prediction", {}).get("primary", {}).get("label", "") for r in results]
            df_results["confidence"] = [r.get("prediction", {}).get("primary", {}).get("confidence", 0) for r in results]
            df_results["confidence_level"] = [r.get("analysis", {}).get("confidence_level", "") for r in results]
            df_results["risk_level"] = [r.get("analysis", {}).get("risk_level", "") for r in results]
            
        elif mode == "expert":
            df_results["predicted_label"] = [r.get("classification", {}).get("primary_prediction", {}).get("label", "") for r in results]
            df_results["confidence"] = [r.get("classification", {}).get("primary_prediction", {}).get("confidence", 0) for r in results]
            df_results["decision_confidence"] = [r.get("classification", {}).get("decision_confidence", "") for r in results]
            df_results["entropy"] = [r.get("probability_analysis", {}).get("entropy", {}).get("normalized", 0) for r in results]

        # Save
        output = BytesIO()
        df_results.to_csv(output, index=False)
        output.seek(0)

        filename = f'predictions_{mode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        return send_file(output, mimetype="text/csv", as_attachment=True, download_name=filename)

    except Exception as e:
        logger.error(f"CSV prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/docs", methods=["GET"])
def docs():
    """API Documentation"""
    return jsonify({
        "api": "Aduan Classification API v5",
        "model": config.HF_MODEL_NAME,
        "labels": LABEL_DESCRIPTIONS,
        "modes": {
            "basic": "Simple prediction with label and confidence",
            "advanced": "Includes secondary predictions, risk level, and confidence analysis",
            "expert": "Detailed analysis with entropy, recommendations, and warnings"
        },
        "endpoints": {
            "POST /predict": "Basic prediction (single text)",
            "POST /predict/advanced": "Advanced prediction (single text)",
            "POST /predict/expert": "Expert prediction (single text)",
            "POST /predict/batch": "Batch prediction (mode: basic/advanced/expert)",
            "POST /predict/csv": "CSV prediction (mode: basic/advanced/expert)",
            "GET /health": "Health check",
            "GET /stats": "API statistics",
            "GET /labels": "Available labels"
        },
        "examples": {
            "basic": {
                "request": {"text": "Ada kebakaran di pasar"},
                "response": {"label": "Darurat", "confidence": 0.95}
            },
            "advanced": {
                "request": {"text": "Ada kebakaran di pasar", "threshold": 0.7},
                "response": {
                    "prediction": {"primary": {"label": "Darurat", "confidence": 0.95}},
                    "analysis": {"confidence_level": "HIGH", "risk_level": "HIGH"}
                }
            },
            "expert": {
                "request": {"text": "Ada kebakaran di pasar"},
                "response": {
                    "classification": {"primary_prediction": {"label": "Darurat"}},
                    "recommendations": {"actions": ["Prioritaskan penanganan segera"]}
                }
            }
        }
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found", "docs": "/docs"}), 404


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal error: {e}")
    return jsonify({"error": "Internal server error"}), 500


# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
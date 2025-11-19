# main.py
"""
FastAPI backend for Karakalpak POS + Morphology multi-task model.
Defaults expect:
 - label_mappings.pkl  (contains pos_label2id, pos_id2label, morph_label2id, morph_id2label)
 - lemma_dict.pkl      (mapping token -> lemma)   <-- your file
 - final_model/        (tokenizer files + optional model_weights.pt)

Save this file alongside label_mappings.pkl and lemma_dict.pkl (or set env vars).
Run with: uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import re
import unicodedata
import pickle
import logging
from typing import List, Optional, Dict, Any

import torch
from torch import nn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, validator
from transformers import AutoTokenizer, XLMRobertaConfig, XLMRobertaModel
import time

# ---------------------------
# CONFIG (change or set env vars)
# ---------------------------
MODEL_DIR = os.environ.get("MODEL_DIR", "final_model7")
LABELS_PATH = os.environ.get("LABELS_PATH", "label_mappings.pkl")
LEMMA_PATH = os.environ.get("LEMMA_PATH", "lemma_dict.pkl")  # uses your file name
BACKBONE = os.environ.get("BACKBONE", "xlm-roberta-base")

# Production security settings
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
MAX_SENTENCE_LENGTH = int(os.environ.get("MAX_SENTENCE_LENGTH", "1000"))
MAX_SENTENCES_PER_REQUEST = int(os.environ.get("MAX_SENTENCES_PER_REQUEST", "50"))
ENABLE_DOCS = os.environ.get("ENABLE_DOCS", "true").lower() == "true"

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("karakalpak_api")

# ---------------------------
# FastAPI
# ---------------------------
app = FastAPI(
    title="Karakalpak POS+Morph API",
    version="1.3",
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    max_age=3600,
)

# GZip compression for responses
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s"
    )
    return response

# ---------------------------
# Pydantic models
# ---------------------------
class PredictRequest(BaseModel):
    sentences: Optional[List[str]] = Field(None, max_items=MAX_SENTENCES_PER_REQUEST)
    sentence: Optional[str] = Field(None, max_length=MAX_SENTENCE_LENGTH)

    @validator('sentences')
    def validate_sentences_length(cls, v):
        if v:
            for sent in v:
                if len(sent) > MAX_SENTENCE_LENGTH:
                    raise ValueError(f"Each sentence must be <= {MAX_SENTENCE_LENGTH} characters")
        return v

    @validator('sentences', 'sentence')
    def validate_at_least_one(cls, v, values):
        if not v and not values.get('sentence') and not values.get('sentences'):
            raise ValueError("Must provide either 'sentence' or 'sentences'")

# ---------------------------
# Globals (populated at startup)
# ---------------------------
tokenizer = None
model = None
device = torch.device("cpu")

pos_label2id: Dict[str, int] = {}
pos_id2label: Dict[int, str] = {}
morph_label2id: Dict[str, Dict[str, int]] = {}
morph_id2label: Dict[str, Dict[int, str]] = {}
morph_features: List[str] = []
lemma_map: Dict[str, str] = {}
lemma_map_norm: Dict[str, str] = {}

# ---------------------------
# Utilities: normalization + pickle loader
# ---------------------------
def load_pickle(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

def normalize_token(tok: str) -> Optional[str]:
    """Normalize token for robust lookup: NFC, lowercase, strip surrounding punctuation."""
    if tok is None:
        return None
    s = unicodedata.normalize("NFC", str(tok))
    s = s.strip()
    # Lowercase (if language-specific casing undesired, remove .lower())
    s = s.lower()
    # Remove leading/trailing punctuation (keep internal punctuation)
    s = re.sub(r'^[\W_]+|[\W_]+$', '', s, flags=re.UNICODE)
    s = re.sub(r'\s+', ' ', s).strip()
    return s if s != "" else None

# ---------------------------
# Model architecture (same as training)
# ---------------------------
class XLMRobertaForMultiTaskTokenClassification(nn.Module):
    def __init__(self, model_checkpoint: str, num_pos_labels: int, num_morph_labels: Dict[str, int]):
        super().__init__()
        self.config = XLMRobertaConfig.from_pretrained(model_checkpoint)
        self.roberta = XLMRobertaModel.from_pretrained(model_checkpoint, config=self.config)
        self.dropout = nn.Dropout(0.1)
        self.pos_classifier = nn.Linear(self.config.hidden_size, num_pos_labels)
        self.morph_classifiers = nn.ModuleDict({
            key: nn.Linear(self.config.hidden_size, num_labels)
            for key, num_labels in num_morph_labels.items()
        })

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        pos_logits = self.pos_classifier(sequence_output)
        morph_logits = {k: cls(sequence_output) for k, cls in self.morph_classifiers.items()}
        out = {"pos_logits": pos_logits}
        out.update({f"{k}_logits": v for k, v in morph_logits.items()})
        return out

# ---------------------------
# Startup: load label mappings, lemma map, tokenizer, model
# ---------------------------
@app.on_event("startup")
def startup():
    global tokenizer, model, device
    global pos_label2id, pos_id2label, morph_label2id, morph_id2label, morph_features
    global lemma_map, lemma_map_norm

    logger.info("Startup: loading label mappings from %s", LABELS_PATH)
    try:
        labels = load_pickle(LABELS_PATH)
    except Exception as e:
        logger.exception("Failed to load label mappings.")
        raise RuntimeError(f"Failed to load {LABELS_PATH}: {e}") from e

    pos_label2id = labels.get("pos_label2id", {})
    pos_id2label = labels.get("pos_id2label", {})
    morph_label2id = labels.get("morph_label2id", {})
    morph_id2label = labels.get("morph_id2label", {})
    morph_features = sorted(list(morph_label2id.keys()))

    logger.info("Loaded labels: POS classes=%d, morph features=%d", len(pos_label2id), len(morph_features))

    # Load lemma map (if exists)
    if os.path.exists(LEMMA_PATH):
        try:
            lemma_map = load_pickle(LEMMA_PATH)
            logger.info("Loaded lemma map (%d entries) from %s", len(lemma_map), LEMMA_PATH)
        except Exception as e:
            logger.exception("Failed to load lemma map; continuing without lemmas.")
            lemma_map = {}
    else:
        logger.info("No lemma file found at %s — lemmas will be null unless fallback used.", LEMMA_PATH)
        lemma_map = {}

    # Build normalized lemma mapping for robust lookup
    lemma_map_norm = {}
    if lemma_map:
        for k, v in lemma_map.items():
            nk = normalize_token(k)
            if nk:
                # keep first occurrence for duplicates
                if nk not in lemma_map_norm:
                    lemma_map_norm[nk] = v
        logger.info("Built normalized lemma map: %d normalized entries (from %d original).",
                    len(lemma_map_norm), len(lemma_map))

    # Tokenizer loading: prefer local MODEL_DIR; avoid hub if local exists
    model_dir_abs = os.path.abspath(MODEL_DIR)
    logger.info("Resolved MODEL_DIR => %s", model_dir_abs)

    try:
        if os.path.isdir(model_dir_abs):
            logger.info("Attempting to load tokenizer from local folder (local_files_only=True)...")
            tokenizer = AutoTokenizer.from_pretrained(model_dir_abs, use_fast=True, local_files_only=True)
            logger.info("Tokenizer loaded from local MODEL_DIR.")
        else:
            logger.warning("MODEL_DIR not found at %s — falling back to backbone tokenizer from hub.", model_dir_abs)
            tokenizer = AutoTokenizer.from_pretrained(BACKBONE, use_fast=True)
            logger.info("Tokenizer loaded from hub backbone: %s", BACKBONE)
    except Exception as e:
        logger.warning("Local tokenizer load failed (%s). Trying hub fallback...", e)
        tokenizer = AutoTokenizer.from_pretrained(BACKBONE, use_fast=True)
        logger.info("Tokenizer loaded from hub: %s", BACKBONE)

    # Build model architecture and try to load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_pos = len(pos_label2id)
    num_morphs = {k: len(v) for k, v in morph_label2id.items()}

    logger.info("Constructing model architecture using backbone: %s", BACKBONE)
    model = XLMRobertaForMultiTaskTokenClassification(BACKBONE, num_pos_labels=num_pos, num_morph_labels=num_morphs)

    # Load weights if available in MODEL_DIR
    state_path = os.path.join(model_dir_abs, "model_weights.pt")
    hf_bin = os.path.join(model_dir_abs, "pytorch_model.bin")

    if os.path.exists(state_path):
        try:
            logger.info("Loading model weights from %s", state_path)
            state = torch.load(state_path, map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state, strict=False)
            logger.info("Loaded model_weights.pt into model (strict=False).")
        except Exception as e:
            logger.exception("Failed to load model_weights.pt; continuing with backbone weights.")
    elif os.path.exists(hf_bin):
        try:
            logger.info("Loading HF-style weights from %s", hf_bin)
            state = torch.load(hf_bin, map_location="cpu")
            model.load_state_dict(state, strict=False)
            logger.info("Loaded pytorch_model.bin into model (strict=False).")
        except Exception as e:
            logger.exception("Failed to load pytorch_model.bin; continuing with backbone weights.")
    else:
        logger.warning("No model weight file found in %s. Classifier heads may be uninitialized.", model_dir_abs)

    model.to(device)
    model.eval()
    logger.info("Startup complete. Device: %s", device)

# ---------------------------
# Prediction core
# ---------------------------
def predict_sentences(sentences: List[str]) -> List[Dict[str, Any]]:
    if not sentences:
        return []

    words_lists = [s.split() for s in sentences]

    encoding = tokenizer(words_lists,
                         is_split_into_words=True,
                         return_tensors="pt",
                         padding=True,
                         truncation=True,
                         max_length=512)

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    pos_preds = outputs["pos_logits"].argmax(dim=-1).cpu()
    morph_preds = {k: outputs[f"{k}_logits"].argmax(dim=-1).cpu() for k in morph_features}

    results = []
    for i in range(input_ids.shape[0]):
        word_ids = encoding.word_ids(batch_index=i)
        sentence_words = words_lists[i]
        preds_for_sentence = []
        prev_wid = None
        for token_pos, wid in enumerate(word_ids):
            if wid is None:
                continue
            if wid != prev_wid:
                try:
                    original_word = sentence_words[wid]
                except Exception:
                    original_word = f"WORD_{wid}"

                # POS
                pos_id = int(pos_preds[i, token_pos].item())
                pos_label = pos_id2label.get(pos_id, str(pos_id))

                # morphology
                morph_out = {}
                for feat in morph_features:
                    pred_id = int(morph_preds[feat][i, token_pos].item())
                    morph_out[feat] = morph_id2label.get(feat, {}).get(pred_id, "-")

                # Lemma lookup: prefer normalized lookup then raw lookup
                lemma_val = None
                try:
                    nk = normalize_token(original_word)
                    if nk and nk in lemma_map_norm:
                        lemma_val = lemma_map_norm[nk]
                    elif original_word in lemma_map:
                        lemma_val = lemma_map[original_word]
                    else:
                        lemma_val = None
                except Exception:
                    lemma_val = None

                preds_for_sentence.append({
                    "word": original_word,
                    "pos": pos_label,
                    "morph": morph_out,
                    "lemma": lemma_val
                })
            prev_wid = wid
        results.append({"input": sentences[i], "words": preds_for_sentence})
    return results

# ---------------------------
# Compactize helper & endpoints
# ---------------------------
def compactize_prediction(pred: Dict[str, Any], lemma_fallback: str = "none") -> Dict[str, Any]:
    words = [w["word"] for w in pred["words"]]
    lemmas = []
    pos_tags = []
    morphology = {feat: [] for feat in morph_features}

    for w in pred["words"]:
        if w.get("lemma") is None:
            if lemma_fallback == "word":
                lm = w["word"]
            elif lemma_fallback == "empty":
                lm = ""
            else:
                lm = None
        else:
            lm = w["lemma"]
        lemmas.append(lm)
        pos_tags.append(w.get("pos"))
        for feat in morph_features:
            morphology[feat].append(w.get("morph", {}).get(feat, "-"))

    return {
        "sentence": pred["input"],
        "words": words,
        "lemmas": lemmas,
        "pos_tags": pos_tags,
        "morphology": morphology
    }

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "name": "Karakalpak POS+Morph API",
        "version": "1.3",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "metadata": "/metadata",
            "predict": "/predict (POST)",
            "predict_compact": "/predict_compact (POST)",
            "docs": "/docs" if ENABLE_DOCS else "disabled"
        }
    }

@app.get("/health")
def health():
    """Health check endpoint for monitoring and Railway"""
    loaded = (tokenizer is not None) and (model is not None)
    labels_loaded = len(pos_label2id) > 0 and len(morph_label2id) > 0

    status = "healthy" if (loaded and labels_loaded) else "unhealthy"

    return {
        "status": status,
        "device": str(device),
        "model_loaded": loaded,
        "labels_loaded": labels_loaded,
        "pos_classes": len(pos_label2id),
        "morph_features": len(morph_features),
        "lemma_entries": len(lemma_map)
    }

@app.get("/metadata")
def metadata():
    """Get model metadata including POS tags and morphological features"""
    return {
        "pos_id2label": pos_id2label,
        "morph_feature_names": morph_features,
        "morph_id2label": morph_id2label
    }

@app.post("/predict")
def predict(req: PredictRequest):
    inputs = []
    if req.sentences:
        inputs = req.sentences
    elif req.sentence:
        inputs = [req.sentence]
    else:
        raise HTTPException(status_code=400, detail="Provide 'sentence' or 'sentences' in request body.")
    try:
        preds = predict_sentences(inputs)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    return preds

@app.post("/predict_compact")
def predict_compact(req: PredictRequest, lemma_fallback: Optional[str] = Query("none",
                                    description="How to fill missing lemmas: 'none' (null), 'word' (use token), 'empty' (empty string)")):
    inputs = []
    if req.sentences:
        inputs = req.sentences
    elif req.sentence:
        inputs = [req.sentence]
    else:
        raise HTTPException(status_code=400, detail="Provide 'sentence' or 'sentences' in request body.")
    if lemma_fallback not in {"none", "word", "empty"}:
        raise HTTPException(status_code=400, detail="lemma_fallback must be one of: none, word, empty")
    try:
        preds = predict_sentences(inputs)
    except Exception as e:
        logger.exception("Compact prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    compact_results = [compactize_prediction(p, lemma_fallback=lemma_fallback) for p in preds]
    return compact_results[0] if len(compact_results) == 1 else compact_results

# ---------------------------
# Run with python main.py for convenience
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)


# uvicorn main:app --reload
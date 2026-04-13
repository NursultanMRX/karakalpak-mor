# main.py
"""
FastAPI backend for Karakalpak POS + Morphology multi-task model.
Optimized for CPU deployment (4 vCPU / 8 GB RAM).

Expects:
 - label_mappings.pkl  (pos_label2id, pos_id2label, morph_label2id, morph_id2label)
 - lemma_dict.pkl      (token -> lemma mapping)
 - final_model7/       (tokenizer files + model.safetensors or model_weights.pt)

Run with: uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import re
import gc
import asyncio
import unicodedata
import pickle
import logging
import time
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from torch import nn
from fastapi import FastAPI, HTTPException, Query, Request, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from transformers import AutoTokenizer, XLMRobertaConfig, XLMRobertaModel
from safetensors.torch import load_file as load_safetensors

# ---------------------------
# CONFIG (change or set env vars)
# ---------------------------
MODEL_DIR = os.environ.get("MODEL_DIR", "final_model7")
LABELS_PATH = os.environ.get("LABELS_PATH", "label_mappings.pkl")
LEMMA_PATH = os.environ.get("LEMMA_PATH", "lemma_dict.pkl")
BACKBONE = os.environ.get("BACKBONE", "xlm-roberta-base")
STATIC_DIR = os.environ.get("STATIC_DIR", "static")

# Production settings
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
MAX_SENTENCE_LENGTH = int(os.environ.get("MAX_SENTENCE_LENGTH", "1000"))
MAX_SENTENCES_PER_REQUEST = int(os.environ.get("MAX_SENTENCES_PER_REQUEST", "50"))
MAX_TEXT_LENGTH = int(os.environ.get("MAX_TEXT_LENGTH", "50000"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
ENABLE_DOCS = os.environ.get("ENABLE_DOCS", "true").lower() == "true"
# Semaphore size: how many inferences run simultaneously
# 4 vCPU / 8 GB: 2 is safe — 2.1 GB model (shared) + 2 × ~400 MB activations ≈ 3 GB peak for ML
INFERENCE_CONCURRENCY = int(os.environ.get("INFERENCE_CONCURRENCY", "2"))
# PyTorch intra-op threads per inference
# Formula: TORCH_THREADS × INFERENCE_CONCURRENCY ≤ vCPUs reserved for ML
# 4 vCPU: 2 threads × 2 concurrent = 4 vCPUs — OS scheduler shares with web/DB at low overhead
TORCH_THREADS = int(os.environ.get("TORCH_THREADS", "2"))
# Max raw request body in bytes (10 MB hard cap to prevent DoS)
MAX_REQUEST_BODY_BYTES = int(os.environ.get("MAX_REQUEST_BODY_BYTES", str(10 * 1024 * 1024)))

# ---------------------------
# CPU optimization: set here after CONFIG so env vars are respected
# 4 vCPU server: 2 threads × Semaphore(2) = 4 vCPUs peak for PyTorch
# OS scheduler naturally yields to web/DB when inference is idle
# ---------------------------
torch.set_num_threads(TORCH_THREADS)
torch.set_num_interop_threads(2)

# API Key Authentication
# Comma-separated list of valid API keys. If empty/unset, auth is DISABLED (open access).
_raw_api_keys = os.environ.get("API_KEYS", "").strip()
API_KEYS: set = set()
if _raw_api_keys:
    API_KEYS = {k.strip() for k in _raw_api_keys.split(",") if k.strip()}
else:
    # Log once at import time so operators notice open access immediately
    import warnings
    warnings.warn(
        "API_KEYS is not set — the API is open to the public with no authentication. "
        "Set API_KEYS env var to enable key-based auth.",
        stacklevel=1,
    )

# Rate limiting: max requests per IP per minute
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "30"))

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("karakalpak_api")

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

# State tracking for /health
_weights_loaded_ok: bool = False
_weight_loading_errors: List[str] = []

# ---------------------------
# POS display names (actual model labels confirmed via /metadata)
# ---------------------------
POS_NAMES: Dict[str, str] = {
    "ALM":     "Almastıq",        # Pronoun
    "ARA_SZ":  "Aralas Sóz",      # Compound / Mixed word
    "ATLQ":    "Atlıq Esim",      # Noun
    "DEM":     "Demonstrativ",    # Demonstrative
    "FYL":     "Feyil",           # Verb
    "JLG":     "Jalǵaw",          # Conjunction
    "JRD_FYL": "Járdemshi Feyil", # Auxiliary verb
    "KBT":     "Kómekshi Bet",    # Postposition
    "RWS":     "Rawısh",          # Adverb
    "SNQ":     "Sanaq",           # Numeral
    "SYM":     "Sımvol",          # Symbol
    "TNS":     "Tańırqaw Sóz",    # Interjection
    "TRK":     "Tirkemes",        # Particle
}

# Concurrency control: INFERENCE_CONCURRENCY inferences run simultaneously
# Memory math (8 GB server): 2.1 GB model (shared) + 2 × ~400 MB activations + OS/web/DB ≈ 4–5 GB peak
_inference_semaphore = asyncio.Semaphore(INFERENCE_CONCURRENCY)

# ---------------------------
# API Key Authentication
# ---------------------------
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(_api_key_header)):
    """Validate API key if API_KEYS is configured. Skip if no keys are set."""
    if not API_KEYS:
        return None  # Auth disabled — open access
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key. Provide X-API-Key header.")
    if api_key not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key.")
    return api_key


# ---------------------------
# Rate Limiting (per-IP, in-memory)
# ---------------------------
_rate_limit_store: Dict[str, List[float]] = {}
_rate_limit_last_cleanup: float = 0.0
_RATE_LIMIT_CLEANUP_INTERVAL = 300.0  # prune stale IPs every 5 minutes


def _check_rate_limit(client_ip: str):
    """Enforce per-IP rate limit. Raises 429 if exceeded."""
    global _rate_limit_last_cleanup

    now = time.time()
    window_start = now - 60.0

    # Periodically remove IPs that have gone idle to prevent unbounded memory growth
    if now - _rate_limit_last_cleanup > _RATE_LIMIT_CLEANUP_INTERVAL:
        stale = [ip for ip, ts in _rate_limit_store.items() if not ts or ts[-1] < window_start]
        for ip in stale:
            del _rate_limit_store[ip]
        _rate_limit_last_cleanup = now

    # Clean old timestamps for this IP and enforce limit
    timestamps = _rate_limit_store.get(client_ip, [])
    timestamps = [t for t in timestamps if t > window_start]

    if len(timestamps) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_PER_MINUTE} requests per minute.",
            headers={"Retry-After": "60"},
        )

    timestamps.append(now)
    _rate_limit_store[client_ip] = timestamps


# ---------------------------
# Utilities
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
    s = s.lower()
    s = re.sub(r'^[\W_]+|[\W_]+$', '', s, flags=re.UNICODE)
    s = re.sub(r'\s+', ' ', s).strip()
    return s if s != "" else None


def split_into_sentences(text: str) -> List[str]:
    """Split raw text into sentences by period, exclamation, and question marks.

    Handles:
    - Decimal numbers: "2.5 km" stays together
    - Multiple spaces/newlines between sentences
    - Empty results are filtered out
    """
    text = text.strip()
    if not text:
        return []

    # Protect decimal numbers by temporarily replacing "digit.digit"
    protected = re.sub(r'(\d)\.(\d)', r'\1<DOT>\2', text)

    # Split on sentence-ending punctuation followed by whitespace or end of string
    parts = re.split(r'(?<=[.!?])\s+', protected)

    sentences = []
    for part in parts:
        restored = part.replace('<DOT>', '.').strip()
        if restored:
            sentences.append(restored)

    return sentences


# ---------------------------
# Model architecture (same as training)
# ---------------------------
class XLMRobertaForMultiTaskTokenClassification(nn.Module):
    def __init__(self, config: XLMRobertaConfig, num_pos_labels: int,
                 num_morph_labels: Dict[str, int]):
        super().__init__()
        self.config = config
        self.roberta = XLMRobertaModel(config)
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
def _load_model_and_data():
    global tokenizer, model, device
    global pos_label2id, pos_id2label, morph_label2id, morph_id2label, morph_features
    global lemma_map, lemma_map_norm
    global _weights_loaded_ok, _weight_loading_errors

    # --- Label mappings ---
    logger.info("Loading label mappings from %s", LABELS_PATH)
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

    logger.info("Loaded labels: POS classes=%d, morph features=%d",
                len(pos_label2id), len(morph_features))

    # --- Lemma map ---
    if os.path.exists(LEMMA_PATH):
        try:
            lemma_map = load_pickle(LEMMA_PATH)
            logger.info("Loaded lemma map (%d entries) from %s",
                        len(lemma_map), LEMMA_PATH)
        except Exception:
            logger.exception("Failed to load lemma map; continuing without lemmas.")
            lemma_map = {}
    else:
        logger.info("No lemma file found at %s — lemmas will be null.", LEMMA_PATH)
        lemma_map = {}

    # Build normalized lemma mapping for robust lookup
    lemma_map_norm = {}
    if lemma_map:
        for k, v in lemma_map.items():
            nk = normalize_token(k)
            if nk and nk not in lemma_map_norm:
                lemma_map_norm[nk] = v
        logger.info("Built normalized lemma map: %d entries (from %d original).",
                     len(lemma_map_norm), len(lemma_map))

    # --- Tokenizer: local only ---
    model_dir_abs = os.path.abspath(MODEL_DIR)
    logger.info("Resolved MODEL_DIR => %s", model_dir_abs)

    if not os.path.isdir(model_dir_abs):
        raise RuntimeError(f"MODEL_DIR not found: {model_dir_abs}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir_abs, use_fast=True, local_files_only=True
        )
        logger.info("Tokenizer loaded from local MODEL_DIR.")
    except Exception as e:
        logger.warning("Local tokenizer load failed (%s). Trying hub fallback...", e)
        tokenizer = AutoTokenizer.from_pretrained(BACKBONE, use_fast=True)
        logger.info("Tokenizer loaded from hub: %s", BACKBONE)

    # --- Construct model from local config (no Hub download) ---
    device = torch.device("cpu")
    num_pos = len(pos_label2id)
    num_morphs = {k: len(v) for k, v in morph_label2id.items()}

    try:
        config = XLMRobertaConfig.from_pretrained(model_dir_abs)
        logger.info("Loaded model config from local MODEL_DIR (no Hub download)")
    except Exception:
        logger.warning("Local config not found, falling back to backbone config")
        config = XLMRobertaConfig.from_pretrained(BACKBONE)

    logger.info("Constructing model architecture...")
    model = XLMRobertaForMultiTaskTokenClassification(
        config, num_pos_labels=num_pos, num_morph_labels=num_morphs
    )

    # --- Load weights: prefer safetensors > .pt > .bin ---
    safetensors_path = os.path.join(model_dir_abs, "model.safetensors")
    pt_path = os.path.join(model_dir_abs, "model_weights.pt")
    hf_bin = os.path.join(model_dir_abs, "pytorch_model.bin")

    state_dict = None
    weight_source = None

    if os.path.exists(safetensors_path):
        logger.info("Loading weights from %s (safetensors - fast & safe)", safetensors_path)
        state_dict = load_safetensors(safetensors_path, device="cpu")
        weight_source = "model.safetensors"
    elif os.path.exists(pt_path):
        logger.info("Loading weights from %s", pt_path)
        state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        weight_source = "model_weights.pt"
    elif os.path.exists(hf_bin):
        logger.info("Loading weights from %s", hf_bin)
        state_dict = torch.load(hf_bin, map_location="cpu", weights_only=True)
        weight_source = "pytorch_model.bin"
    else:
        raise RuntimeError(
            f"No model weight file found in {model_dir_abs}. "
            f"Expected model.safetensors, model_weights.pt, or pytorch_model.bin"
        )

    # --- Load state dict with diagnostics ---
    result = model.load_state_dict(state_dict, strict=False)

    if result.unexpected_keys:
        logger.warning("Unexpected keys in %s (ignored): %d keys",
                        weight_source, len(result.unexpected_keys))
    if result.missing_keys:
        logger.warning("Missing keys in %s: %d keys", weight_source, len(result.missing_keys))

    # Classify missing keys
    missing_backbone = [k for k in result.missing_keys if k.startswith("roberta.")]
    missing_heads = [k for k in result.missing_keys if not k.startswith("roberta.")]

    if missing_backbone:
        raise RuntimeError(
            f"FATAL: {len(missing_backbone)} backbone keys missing from {weight_source}. "
            f"Model is unusable. Examples: {missing_backbone[:5]}"
        )

    if missing_heads:
        _weight_loading_errors.append(
            f"{len(missing_heads)} classifier head keys missing: {missing_heads}"
        )
        logger.error(
            "%d classifier head keys not loaded (predictions may be unreliable): %s",
            len(missing_heads), missing_heads
        )

    _weights_loaded_ok = len(result.missing_keys) == 0
    del state_dict

    # --- INT8 dynamic quantization for CPU performance ---
    logger.info("Applying INT8 dynamic quantization to nn.Linear layers...")
    model_q = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    model = model_q

    model.to(device)
    model.eval()

    # Free transient memory
    gc.collect()

    logger.info(
        "Startup complete. Device: %s | Weights: %s | Quantized: INT8 | Status: %s",
        device, weight_source, "OK" if _weights_loaded_ok else "DEGRADED"
    )


# ---------------------------
# Lifespan context manager (replaces deprecated @app.on_event)
# ---------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model_and_data()
    yield
    logger.info("Shutting down.")


# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(
    title="Karakalpak POS+Morph API",
    version="1.5",
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None,
    lifespan=lifespan,
)

# CORS middleware (credentials disabled with wildcard origin per spec)
_allow_credentials = "*" not in ALLOWED_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=_allow_credentials,
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
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# Request body size limit middleware (DoS protection — reject oversized bodies early)
@app.middleware("http")
async def limit_request_body(request: Request, call_next):
    if request.method in ("POST", "PUT", "PATCH"):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request body too large. Maximum {MAX_REQUEST_BODY_BYTES} bytes."},
            )
    return await call_next(request)


# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Skip rate limiting for health checks and docs
    skip_paths = {"/health", "/", "/docs", "/redoc", "/openapi.json"}
    if request.url.path not in skip_paths:
        client_ip = request.client.host if request.client else "unknown"
        try:
            _check_rate_limit(client_ip)
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
                headers=exc.headers or {},
            )
    response = await call_next(request)
    return response


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        "%s %s - Status: %s - Time: %.3fs",
        request.method, request.url.path, response.status_code, process_time
    )
    return response


# ---------------------------
# Pydantic models
# ---------------------------
class PredictRequest(BaseModel):
    sentences: Optional[List[str]] = Field(None)
    sentence: Optional[str] = Field(None, max_length=MAX_SENTENCE_LENGTH)
    text: Optional[str] = Field(
        None, max_length=MAX_TEXT_LENGTH,
        description="Raw text to auto-split into sentences by '.' and analyze in batches"
    )

    @field_validator('sentences')
    @classmethod
    def validate_sentences(cls, v):
        if v is not None:
            if len(v) > MAX_SENTENCES_PER_REQUEST:
                raise ValueError(
                    f"Maximum {MAX_SENTENCES_PER_REQUEST} sentences allowed, got {len(v)}"
                )
            for i, sent in enumerate(v):
                if len(sent) > MAX_SENTENCE_LENGTH:
                    raise ValueError(
                        f"Sentence {i} exceeds max length {MAX_SENTENCE_LENGTH}"
                    )
                if not sent.strip():
                    raise ValueError(f"Sentence {i} is empty or whitespace-only")
        return v

    @field_validator('sentence')
    @classmethod
    def validate_sentence_not_empty(cls, v):
        if v is not None and not v.strip():
            raise ValueError("'sentence' must not be empty or whitespace-only")
        return v

    @field_validator('text')
    @classmethod
    def validate_text_not_empty(cls, v):
        if v is not None and not v.strip():
            raise ValueError("'text' must not be empty or whitespace-only")
        return v

    @model_validator(mode='after')
    def validate_at_least_one(self):
        if self.sentences is None and self.sentence is None and self.text is None:
            raise ValueError(
                "Must provide at least one of 'sentence', 'sentences', or 'text'"
            )
        return self


class AnalyzeRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=MAX_TEXT_LENGTH,
        description="Raw text to split into sentences and analyze"
    )

    @field_validator('text')
    @classmethod
    def validate_text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("'text' must not be empty or whitespace-only")
        return v


# ---------------------------
# Prediction core (batched for memory efficiency)
# ---------------------------
def _predict_batch(sentences: List[str]) -> List[Dict[str, Any]]:
    """Run inference on a single batch of sentences."""
    words_lists = [s.split() for s in sentences]

    encoding = tokenizer(
        words_lists,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.inference_mode():
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
                except IndexError:
                    original_word = f"WORD_{wid}"

                # POS
                pos_id = int(pos_preds[i, token_pos].item())
                pos_label = pos_id2label.get(pos_id, str(pos_id))

                # Morphology
                morph_out = {}
                for feat in morph_features:
                    pred_id = int(morph_preds[feat][i, token_pos].item())
                    morph_out[feat] = morph_id2label.get(feat, {}).get(pred_id, "-")

                # Lemma lookup: prefer normalized, then raw
                lemma_val = None
                nk = normalize_token(original_word)
                if nk and nk in lemma_map_norm:
                    lemma_val = lemma_map_norm[nk]
                elif original_word in lemma_map:
                    lemma_val = lemma_map[original_word]

                preds_for_sentence.append({
                    "word": original_word,
                    "pos": pos_label,
                    "morph": morph_out,
                    "lemma": lemma_val,
                })
            prev_wid = wid

        results.append({"input": sentences[i], "words": preds_for_sentence})

    return results


def predict_sentences_batched(sentences: List[str]) -> List[Dict[str, Any]]:
    """Process sentences in batches of BATCH_SIZE to control memory usage."""
    if not sentences:
        return []

    all_results = []
    for batch_start in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[batch_start:batch_start + BATCH_SIZE]
        batch_results = _predict_batch(batch)
        all_results.extend(batch_results)

    return all_results


async def _run_inference(sentences: List[str]) -> List[Dict[str, Any]]:
    """Run inference with concurrency control (semaphore prevents OOM)."""
    async with _inference_semaphore:
        return await asyncio.to_thread(predict_sentences_batched, sentences)


# ---------------------------
# Compactize helper
# ---------------------------
def compactize_prediction(
    pred: Dict[str, Any],
    lemma_fallback: str = "none",
    allowed_morph: Optional[set] = None,  # None = all features; set() = none; {a,b} = subset
) -> Dict[str, Any]:
    active_feats = morph_features if allowed_morph is None else [f for f in morph_features if f in allowed_morph]

    words = [w["word"] for w in pred["words"]]
    lemmas = []
    pos_tags = []
    morphology: Dict[str, List] = {feat: [] for feat in active_feats}

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
        for feat in active_feats:
            morphology[feat].append(w.get("morph", {}).get(feat, "-"))

    return {
        "sentence":  pred["input"],
        "words":     words,
        "lemmas":    lemmas,
        "pos_tags":  pos_tags,
        "morphology": morphology,
    }


# ---------------------------
# Response shaping helpers
# ---------------------------
def _parse_morph_fields(morph_fields: Optional[str]) -> Optional[set]:
    """
    Parse the morph_fields query param into a set of allowed feature names.
      'all'  (or None) → None  (keep everything — fast path)
      'none'           → empty set (strip all morph)
      'a,b,c'          → {'a','b','c'} intersected with known features
    """
    if not morph_fields or morph_fields == "all":
        return None
    if morph_fields == "none":
        return set()
    return {f.strip() for f in morph_fields.split(",") if f.strip()} & set(morph_features)


def _apply_response_options(
    results: List[Dict[str, Any]],
    allowed_fields: Optional[set],  # None = keep all; set() = strip all; {a,b} = keep only a,b
    include_pos_name: bool,
    lemma_fallback: str = "none",
) -> List[Dict[str, Any]]:
    """
    Post-process raw prediction results:
      - Filter morph features to only those in allowed_fields
      - Optionally inject pos_name (human-readable POS label)
      - Apply lemma fallback for null lemmas
    This runs after inference so the core inference path is never changed.
    """
    # Fast path: nothing to change
    if allowed_fields is None and not include_pos_name and lemma_fallback == "none":
        return results

    processed = []
    for sent in results:
        new_words = []
        for w in sent["words"]:
            new_w: Dict[str, Any] = {
                "word": w["word"],
                "pos":  w["pos"],
            }
            if include_pos_name:
                new_w["pos_name"] = POS_NAMES.get(w["pos"], w["pos"])
            # Lemma with fallback
            lemma = w.get("lemma")
            if lemma is None:
                if lemma_fallback == "word":
                    lemma = w["word"]
                elif lemma_fallback == "empty":
                    lemma = ""
            new_w["lemma"] = lemma
            # Morph filter
            if allowed_fields is None:
                new_w["morph"] = w["morph"]
            else:
                new_w["morph"] = {k: v for k, v in w["morph"].items() if k in allowed_fields}
            new_words.append(new_w)
        processed.append({"input": sent["input"], "words": new_words})
    return processed


# ---------------------------
# Helper: extract sentences from request
# ---------------------------
def _extract_sentences(req: PredictRequest) -> List[str]:
    """Extract or split sentences from request. Priority: text > sentences > sentence."""
    if req.text:
        return split_into_sentences(req.text)
    elif req.sentences:
        return req.sentences
    elif req.sentence:
        return [req.sentence]
    else:
        raise HTTPException(status_code=400, detail="No input provided.")


# ---------------------------
# Endpoints
# ---------------------------
@app.get("/")
def root():
    # Serve frontend if built, otherwise return API info
    index_file = Path(STATIC_DIR) / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "name": "Karakalpak POS+Morph API",
        "version": "1.6",
        "status": "running",
        "auth_enabled": len(API_KEYS) > 0,
        "rate_limit": f"{RATE_LIMIT_PER_MINUTE} requests/minute",
        "endpoints": {
            "health": "/health",
            "metadata": "/metadata",
            "predict":         "/predict (POST)",
            "predict_compact": "/predict_compact (POST)",
            "analyze":         "/analyze (POST)",
            "words":           "/words (POST) — flat word list, best for frontend tables",
            "docs":            "/docs" if ENABLE_DOCS else "disabled",
        },
    }


@app.get("/health")
def health():
    loaded = (tokenizer is not None) and (model is not None)
    labels_loaded = len(pos_label2id) > 0 and len(morph_label2id) > 0

    if loaded and labels_loaded and _weights_loaded_ok:
        status = "healthy"
    elif loaded and labels_loaded:
        status = "degraded"
    else:
        status = "unhealthy"

    result = {
        "status": status,
        "device": str(device),
        "model_loaded": loaded,
        "labels_loaded": labels_loaded,
        "weights_ok": _weights_loaded_ok,
        "quantized": "INT8",
        "pos_classes": len(pos_label2id),
        "morph_features": len(morph_features),
        "lemma_entries": len(lemma_map),
        "batch_size": BATCH_SIZE,
    }

    if _weight_loading_errors:
        result["weight_warnings"] = _weight_loading_errors

    return result


@app.get("/metadata")
def metadata():
    return {
        "pos_id2label":        pos_id2label,
        "pos_names":           POS_NAMES,
        "morph_feature_names": morph_features,
        "morph_id2label":      morph_id2label,
        "note": (
            "All 25 morph features are predicted for every word — "
            "the model has no 'not applicable' class. "
            "Use ?morph_fields= on prediction endpoints to select only the features you need."
        ),
    }


@app.post("/predict")
async def predict(
    req: PredictRequest,
    morph_fields: Optional[str] = Query(
        "all",
        description=(
            "Morph features to include. Options: "
            "'all' (default — all 25 features), "
            "'none' (strip morph entirely), "
            "or comma-separated feature names e.g. 'zaman,bet,seplik,san'"
        ),
    ),
    include_pos_name: bool = Query(
        False,
        description="Add 'pos_name' field with human-readable POS label alongside the code",
    ),
    _key: str = Depends(verify_api_key),
):
    inputs = _extract_sentences(req)
    if not inputs:
        raise HTTPException(status_code=400, detail="No sentences found in input.")
    try:
        preds = await _run_inference(inputs)
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal prediction error.")
    return _apply_response_options(preds, _parse_morph_fields(morph_fields), include_pos_name)


@app.post("/predict_compact")
async def predict_compact(
    req: PredictRequest,
    lemma_fallback: Optional[str] = Query(
        "none",
        description="How to fill missing lemmas: 'none' (null), 'word' (use token), 'empty' ('')",
    ),
    morph_fields: Optional[str] = Query(
        "all",
        description="Morph features to include: 'all', 'none', or 'feat1,feat2,...'",
    ),
    _key: str = Depends(verify_api_key),
):
    inputs = _extract_sentences(req)
    if not inputs:
        raise HTTPException(status_code=400, detail="No sentences found in input.")
    if lemma_fallback not in {"none", "word", "empty"}:
        raise HTTPException(
            status_code=400, detail="lemma_fallback must be one of: none, word, empty"
        )
    try:
        preds = await _run_inference(inputs)
    except Exception:
        logger.exception("Compact prediction failed")
        raise HTTPException(status_code=500, detail="Internal prediction error.")

    allowed = _parse_morph_fields(morph_fields)
    compact_results = [
        compactize_prediction(p, lemma_fallback=lemma_fallback, allowed_morph=allowed)
        for p in preds
    ]
    return compact_results


@app.post("/analyze")
async def analyze(
    req: AnalyzeRequest,
    morph_fields: Optional[str] = Query(
        "all",
        description="Morph features to include: 'all', 'none', or 'feat1,feat2,...'",
    ),
    include_pos_name: bool = Query(
        False,
        description="Add 'pos_name' field with human-readable POS label",
    ),
    _key: str = Depends(verify_api_key),
):
    """Analyze raw text: split into sentences, process in batches, return unified results."""
    sentences = split_into_sentences(req.text)
    if not sentences:
        raise HTTPException(
            status_code=400, detail="No sentences found in text after splitting."
        )
    try:
        preds = await _run_inference(sentences)
    except Exception:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail="Internal prediction error.")
    preds = _apply_response_options(preds, _parse_morph_fields(morph_fields), include_pos_name)
    return {
        "text": req.text,
        "sentence_count": len(sentences),
        "sentences": preds,
    }


@app.post("/words")
async def words_flat(
    req: AnalyzeRequest,
    morph_fields: Optional[str] = Query(
        "all",
        description=(
            "Morph features to include: 'all', 'none', or comma-separated names "
            "e.g. 'zaman,bet,seplik,san,feyil_formasi'"
        ),
    ),
    include_pos_name: bool = Query(
        True,
        description="Include human-readable 'pos_name' field (default: true)",
    ),
    lemma_fallback: Optional[str] = Query(
        "word",
        description="How to fill missing lemmas: 'none' (null), 'word' (use original token), 'empty' ('')",
    ),
    _key: str = Depends(verify_api_key),
):
    """
    Frontend-optimised endpoint. Returns a flat list of every word across all sentences —
    no nested loops needed. Ideal for word-by-word tables, annotation UIs, and exports.

    Each word includes its sentence_index so the frontend can group by sentence if needed.
    Defaults: pos_name=true, lemma_fallback='word' (no null lemmas by default).
    """
    sentences = split_into_sentences(req.text)
    if not sentences:
        raise HTTPException(status_code=400, detail="No sentences found in text after splitting.")
    if lemma_fallback not in {"none", "word", "empty"}:
        raise HTTPException(status_code=400, detail="lemma_fallback must be one of: none, word, empty")

    try:
        preds = await _run_inference(sentences)
    except Exception:
        logger.exception("Words endpoint failed")
        raise HTTPException(status_code=500, detail="Internal prediction error.")

    preds = _apply_response_options(
        preds,
        _parse_morph_fields(morph_fields),
        include_pos_name,
        lemma_fallback=lemma_fallback,
    )

    flat_words = []
    for sent_idx, sent in enumerate(preds):
        for word_idx, w in enumerate(sent["words"]):
            entry: Dict[str, Any] = {
                "sentence_index": sent_idx,
                "word_index":     word_idx,
                "word":           w["word"],
                "pos":            w["pos"],
            }
            if include_pos_name:
                entry["pos_name"] = w.get("pos_name", w["pos"])
            entry["lemma"] = w["lemma"]
            entry["morph"] = w["morph"]
            flat_words.append(entry)

    return {
        "text":           req.text,
        "sentence_count": len(sentences),
        "word_count":     len(flat_words),
        "words":          flat_words,
    }


# ---------------------------
# Static file serving for frontend
# MUST be last — catch-all /{path:path} would intercept API routes if registered early
# ---------------------------
static_path = Path(STATIC_DIR)

@app.get("/{path:path}")
async def serve_static_files(path: str):
    """Serve static files (CSS, JS, etc.)"""
    file_path = static_path / path
    if file_path.exists() and file_path.is_file():
        return FileResponse(str(file_path))
    # Fall through to 404 — don't expose internal paths
    raise HTTPException(status_code=404, detail="Not found")


# ---------------------------
# Run with python main.py for convenience
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

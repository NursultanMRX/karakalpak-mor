# Karakalpak POS + Morphological Analysis API

FastAPI backend for Karakalpak language NLP: POS tagging, morphological analysis, and lemmatization using a fine-tuned XLM-RoBERTa-base model.

- **13 POS classes** + **25 morphological features** per token
- CPU-only, INT8 dynamic quantization at runtime
- Optimized for **4 vCPU / 8 GB RAM** VPS
- Model hosted on Hugging Face — not baked into the Docker image (~340 MB image)

---

## VPS Deployment (Docker)

### Requirements

- Ubuntu/Debian VPS: 4 vCPU / 8 GB RAM minimum
- Docker + Docker Compose installed
- Python 3.8+ (for the one-time model download)
- Hugging Face READ token for `nickoo004/karakalpak-pos-morph-model`

---

### Step 1 — Clone the repo

```bash
git clone https://github.com/NursultanMRX/karakalpak-mor.git /opt/karakalpak
cd /opt/karakalpak
```

---

### Step 2 — Download the model (one-time, ~1.06 GB)

The Docker image does **not** contain the model weights. Download them once to disk:

```bash
pip3 install huggingface_hub
python3 download_model.py --token hf_YOUR_READ_TOKEN --dest /opt/karakalpak-model
```

Expected output:
```
  [1/8] Downloading model.safetensors...
         Done (1063.4 MB).
  [2/8] Downloading config.json...
         Done (0.0 MB).
  ...
  [8/8] Downloading lemma_dict.pkl...
         Done (0.3 MB).

All files ready in: /opt/karakalpak-model
```

Re-running the script is safe — it skips files that already exist.

---

### Step 3 — Configure environment

```bash
cp .env.example .env
nano .env
```

Minimum settings to configure:

| Variable | Description | Example |
|----------|-------------|---------|
| `API_KEYS` | Comma-separated auth keys. Empty = no auth (not recommended) | `kaa_abc123,kaa_xyz456` |
| `ALLOWED_ORIGINS` | CORS. Use `*` for open or your frontend domain | `https://yoursite.com` |
| `RATE_LIMIT_PER_MINUTE` | Max requests per IP per minute | `30` |

Generate a secure API key:
```bash
python3 -c "import secrets; print('kaa_' + secrets.token_urlsafe(32))"
```

---

### Step 4 — Pull the Docker image

```bash
docker pull nickoo004/karakalpak-api:latest
```

Or build from source:
```bash
docker build -t karakalpak-api:latest .
# Then update image name in docker-compose.yml to karakalpak-api:latest
```

---

### Step 5 — Start the service

```bash
docker compose up -d
```

Watch startup — the model takes **60–90 seconds** to load:
```bash
docker compose logs -f
```

You should see:
```
INFO: Model loaded successfully in 72.3s
INFO: Application startup complete.
```

---

### Step 6 — Verify

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "ok", "model_loaded": true, "version": "1.6"}
```

---

### Step 7 — Nginx reverse proxy (HTTPS)

```nginx
server {
    listen 443 ssl;
    server_name api.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 60s;
    }
}
```

Get a free SSL certificate:
```bash
apt install certbot python3-certbot-nginx
certbot --nginx -d api.yourdomain.com
```

---

## Updating

```bash
cd /opt/karakalpak
git pull
docker pull nickoo004/karakalpak-api:latest
docker compose up -d --force-recreate
```

Model files in `/opt/karakalpak-model` are **not** affected by image updates.

---

## API Reference

### Authentication

All `POST` endpoints require the `X-API-Key` header when `API_KEYS` is set:

```
X-API-Key: kaa_YOUR_KEY
```

---

### Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | No | API info and version |
| GET | `/health` | No | Health check + model status |
| GET | `/metadata` | No | POS labels, morph features, model info |
| POST | `/predict` | Yes* | Analyze one or more sentences |
| POST | `/predict_compact` | Yes* | Compact format (smaller response) |
| POST | `/analyze` | Yes* | Full text with auto sentence splitting |
| POST | `/words` | Yes* | **Flat word list — best for frontend** |

*Auth required only when `API_KEYS` env var is set.

---

### POST `/words` — Flat word list (recommended for frontend)

**Query parameters:**

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `morph_fields` | `all` | `all`, `none`, `Case,Number,...` | Which morph features to include |
| `include_pos_name` | `true` | `true`/`false` | Include human-readable POS name |
| `lemma_fallback` | `word` | `word`, `none` | If lemma missing: return original word or null |

**Request:**
```bash
curl -X POST "http://localhost:8000/words?morph_fields=Case,Number&include_pos_name=true" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: kaa_YOUR_KEY" \
  -d '{"text": "Men mektepke baraman. Ol kitap oqıydı."}'
```

**Response:**
```json
{
  "text": "Men mektepke baraman. Ol kitap oqıydı.",
  "sentence_count": 2,
  "word_count": 5,
  "words": [
    {
      "sentence_index": 0,
      "word_index": 0,
      "word": "Men",
      "pos": "ALM",
      "pos_name": "Almastıq",
      "lemma": "men",
      "morph": {"Case": "Nom", "Number": "Sing"}
    },
    {
      "sentence_index": 0,
      "word_index": 1,
      "word": "mektepke",
      "pos": "ATLQ",
      "pos_name": "Atlıq Esim",
      "lemma": "mektep",
      "morph": {"Case": "Dat", "Number": "Sing"}
    }
  ]
}
```

---

### POST `/predict` — Full prediction

```bash
# Single sentence
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: kaa_YOUR_KEY" \
  -d '{"sentence": "Men mektepke baraman"}'

# Multiple sentences
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: kaa_YOUR_KEY" \
  -d '{"sentences": ["Men mektepke baraman", "Ol kitap oqıydı"]}'

# Auto-split large text
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: kaa_YOUR_KEY" \
  -d '{"text": "Men mektepke baraman. Ol kitap oqıydı."}'
```

---

### JavaScript (Frontend)

```javascript
const res = await fetch("https://api.yourdomain.com/words?morph_fields=Case,Number", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": "kaa_YOUR_KEY"
  },
  body: JSON.stringify({ text: "Men mektepke baraman. Ol kitap oqıydı." })
});
const data = await res.json();
// data.words → flat array of all words across all sentences
```

---

## POS Classes

| Code | Karakalpak | Meaning |
|------|------------|---------|
| ALM | Almastıq | Pronoun |
| ARA_SZ | Aralas Sóz | Conjunction |
| ATLQ | Atlıq Esim | Noun |
| DEM | Demonstrativ | Demonstrative |
| FYL | Feyil | Verb |
| JLG | Jalǵaw | Particle |
| JRD_FYL | Járdemshi Feyil | Auxiliary Verb |
| KBT | Kómekshi Bet | Auxiliary |
| RWS | Rawısh | Adverb |
| SNQ | Sanaq | Numeral |
| SYM | Sımvol | Symbol |
| TNS | Tańırqaw Sóz | Interjection |
| TRK | Tirkemes | Adposition |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `final_model7` | Model directory path |
| `LABELS_PATH` | `label_mappings.pkl` | Label mappings file |
| `LEMMA_PATH` | `lemma_dict.pkl` | Lemma dictionary file |
| `ALLOWED_ORIGINS` | `*` | CORS origins (comma-separated) |
| `API_KEYS` | *(empty)* | Auth keys. Empty = auth disabled |
| `RATE_LIMIT_PER_MINUTE` | `30` | Max requests per IP per minute |
| `MAX_SENTENCE_LENGTH` | `1000` | Max characters per sentence |
| `MAX_SENTENCES_PER_REQUEST` | `50` | Max sentences per request |
| `MAX_TEXT_LENGTH` | `50000` | Max characters for `text` field |
| `BATCH_SIZE` | `4` | Sentences per inference batch |
| `INFERENCE_CONCURRENCY` | `2` | Simultaneous inference slots |
| `TORCH_THREADS` | `2` | PyTorch threads per inference |
| `ENABLE_DOCS` | `true` | Enable Swagger UI at `/docs` |

---

## Performance (4 vCPU / 8 GB RAM)

| Metric | Value |
|--------|-------|
| Model memory (idle) | ~2.1 GB |
| Peak inference memory | ~3.0 GB |
| Available for OS + DB | ~5 GB |
| Concurrent users | 4–5 without queuing |
| Single sentence latency | ~0.8–1.2s |

---

## Troubleshooting

**Container exits at startup:**
```bash
docker compose logs karakalpak-api
```
Most likely cause: model files not mounted. Verify `/opt/karakalpak-model` has all 8 files.

**Out of memory:**
Lower concurrency in `.env`:
```
INFERENCE_CONCURRENCY=1
BATCH_SIZE=2
```
Then: `docker compose up -d`

**Port already in use:**
Edit `docker-compose.yml` — change `"8000:8000"` to `"8080:8000"`.

---

## Project Structure

```
.
├── main.py                 # All API code (~850 lines)
├── requirements.txt        # Python dependencies
├── Dockerfile              # Python 3.11-slim, non-root user
├── docker-compose.yml      # VPS deployment config
├── download_model.py       # Download model from HF to VPS disk
├── upload_to_hf.py         # Upload model to HF (dev use only)
├── .env.example            # Environment variable template
└── railway.toml            # Railway.app config (alternative deployment)
```

Model files are **not** in this repo. They are stored at:
`nickoo004/karakalpak-pos-morph-model` on Hugging Face (private).

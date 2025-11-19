# Karakalpak POS+Morph API

A production-ready FastAPI backend for Karakalpak language NLP, providing part-of-speech tagging, morphological analysis, and lemmatization using a fine-tuned XLM-RoBERTa model.

## Features

- **Multi-task NLP**: POS tagging, morphological feature extraction, and lemmatization
- **Production-ready**: CORS, compression, request logging, input validation
- **Flexible API**: Single sentence or batch processing
- **Railway-optimized**: Ready for one-click deployment

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check for monitoring
- `GET /metadata` - Model metadata (POS tags, morphological features)
- `POST /predict` - Full prediction with detailed output
- `POST /predict_compact` - Compact prediction format

## Quick Start

### Prerequisites

- Python 3.9+
- Model files:
  - `label_mappings.pkl` - POS and morphology label mappings
  - `lemma_dict.pkl` - Token-to-lemma dictionary
  - `final_model7/` - Trained model directory with tokenizer

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run the server**
   ```bash
   uvicorn main:app --reload
   ```

   The API will be available at `http://localhost:8000`

## Deployment to Railway

### Method 1: GitHub Integration (Recommended)

1. **Prepare your repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Production-ready Karakalpak API"
   ```

2. **Create GitHub repository**
   - Go to [GitHub](https://github.com/new)
   - Create a new repository
   - Push your code:
     ```bash
     git remote add origin <your-github-repo-url>
     git branch -M main
     git push -u origin main
     ```

3. **Deploy on Railway**
   - Go to [Railway](https://railway.app)
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway will automatically detect the Python project

4. **Configure environment variables in Railway**
   - Go to your project settings
   - Add variables from `.env.example`:
     - `MODEL_DIR=final_model7`
     - `LABELS_PATH=label_mappings.pkl`
     - `LEMMA_PATH=lemma_dict.pkl`
     - `ALLOWED_ORIGINS=https://yourfrontend.com` (or `*` for testing)
     - `ENABLE_DOCS=false` (for production)

5. **Important: Model Files**

   Your model files are too large for Git. You have two options:

   **Option A: Use Git LFS (Large File Storage)**
   ```bash
   git lfs install
   git lfs track "*.pkl"
   git lfs track "*.pt"
   git lfs track "*.bin"
   git add .gitattributes
   git add label_mappings.pkl lemma_dict.pkl final_model7/
   git commit -m "Add model files with Git LFS"
   git push
   ```

   **Option B: Upload via Railway Volume (Recommended for very large models)**
   - Remove model files from git (keep them in .gitignore)
   - Use Railway's persistent volumes or object storage
   - Upload files manually to Railway volume
   - Update `MODEL_DIR`, `LABELS_PATH`, `LEMMA_PATH` to point to volume

### Method 2: Railway CLI

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Deploy
railway up
```

## Environment Variables

See `.env.example` for all available configuration options:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_DIR` | Model directory path | `final_model7` |
| `LABELS_PATH` | Label mappings file | `label_mappings.pkl` |
| `LEMMA_PATH` | Lemma dictionary file | `lemma_dict.pkl` |
| `BACKBONE` | HuggingFace model name | `xlm-roberta-base` |
| `ALLOWED_ORIGINS` | CORS allowed origins | `*` |
| `MAX_SENTENCE_LENGTH` | Max sentence characters | `1000` |
| `MAX_SENTENCES_PER_REQUEST` | Max batch size | `50` |
| `ENABLE_DOCS` | Enable API docs | `true` |

## API Usage Examples

### Single Sentence Prediction

```bash
curl -X POST "https://your-app.railway.app/predict" \
  -H "Content-Type: application/json" \
  -d '{"sentence": "Менің атым Айгүл"}'
```

### Batch Prediction

```bash
curl -X POST "https://your-app.railway.app/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sentences": [
      "Менің атым Айгүл",
      "Мен студентпін"
    ]
  }'
```

### Compact Format

```bash
curl -X POST "https://your-app.railway.app/predict_compact?lemma_fallback=word" \
  -H "Content-Type: application/json" \
  -d '{"sentence": "Менің атым Айгүл"}'
```

## Security Features

- **CORS Configuration**: Configurable allowed origins
- **Input Validation**: Max sentence length and batch size limits
- **Request Logging**: All requests logged with timing
- **GZip Compression**: Automatic response compression
- **Rate Limiting**: Consider adding `slowapi` for production rate limits

## Production Checklist

- [ ] Model files uploaded and accessible
- [ ] Environment variables configured in Railway
- [ ] `ALLOWED_ORIGINS` set to your frontend domain (not `*`)
- [ ] `ENABLE_DOCS` set to `false` (or protect with auth)
- [ ] Health checks working (`/health` endpoint)
- [ ] Custom domain configured (optional)
- [ ] Monitoring set up (Railway provides basic metrics)
- [ ] Consider adding authentication for production use

## Monitoring

Railway provides built-in monitoring:
- View logs: `railway logs`
- Check metrics: Railway dashboard
- Health endpoint: `GET /health`

## Troubleshooting

### Model files not found
- Verify files are uploaded to Railway
- Check environment variables point to correct paths
- Ensure Git LFS is properly configured if using it

### Out of memory errors
- Upgrade Railway plan for more RAM
- Consider using CPU-only PyTorch build
- Reduce batch size (`MAX_SENTENCES_PER_REQUEST`)

### Startup timeout
- Railway default timeout is 100s
- Model loading may take time
- Consider using persistent volumes for faster restarts

## Development

### Running Tests
```bash
pytest  # Add tests in tests/ directory
```

### Code Quality
```bash
# Format code
black main.py

# Lint
flake8 main.py
```

## License

[Your License Here]

## Support

For issues and questions:
- Open an issue on GitHub
- Check Railway documentation: https://docs.railway.app

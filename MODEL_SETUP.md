# Model Files Setup Guide

This guide helps you manage and deploy the large model files required by the Karakalpak API.

## Required Files

Your application needs these files to function:

1. **label_mappings.pkl** (~KB) - POS and morphology label mappings
2. **lemma_dict.pkl** (~MB) - Token-to-lemma dictionary
3. **final_model7/** directory containing:
   - `config.json` - Model configuration
   - `tokenizer.json` - Tokenizer configuration
   - `tokenizer_config.json` - Tokenizer settings
   - `special_tokens_map.json` - Special tokens
   - `model_weights.pt` or `pytorch_model.bin` (~GB) - Trained weights

## Deployment Options

### Option 1: Git LFS (for files < 2GB)

**Best for**: Model files under 2GB, automated deployments

1. **Install Git LFS**
   ```bash
   # On Windows
   git lfs install

   # On macOS
   brew install git-lfs
   git lfs install

   # On Linux
   sudo apt-get install git-lfs
   git lfs install
   ```

2. **Track large files** (already configured in .gitattributes)
   ```bash
   git lfs track "*.pkl"
   git lfs track "*.pt"
   git lfs track "*.bin"
   ```

3. **Add and commit files**
   ```bash
   git add .gitattributes
   git add label_mappings.pkl lemma_dict.pkl
   git add final_model7/
   git commit -m "Add model files via Git LFS"
   git push origin main
   ```

4. **Verify LFS tracking**
   ```bash
   git lfs ls-files
   ```

**Note**: GitHub has LFS bandwidth limits. Consider GitHub Pro or use Railway volumes for very large models.

### Option 2: Railway Volumes (for files > 2GB)

**Best for**: Very large models, frequently updated models

1. **Keep model files in .gitignore** (they're already there)

2. **Create a Railway volume**
   - In Railway dashboard, go to your service
   - Click "Variables" → "Add Volume"
   - Mount path: `/app/models`

3. **Upload files to Railway volume**
   ```bash
   # Using Railway CLI
   railway volumes upload /app/models label_mappings.pkl
   railway volumes upload /app/models lemma_dict.pkl
   railway volumes upload /app/models/final_model7 final_model7/
   ```

4. **Update environment variables in Railway**
   ```
   MODEL_DIR=/app/models/final_model7
   LABELS_PATH=/app/models/label_mappings.pkl
   LEMMA_PATH=/app/models/lemma_dict.pkl
   ```

### Option 3: External Storage (S3, Google Cloud Storage)

**Best for**: Multiple deployments, CI/CD pipelines

1. **Upload to cloud storage**
   - Use AWS S3, Google Cloud Storage, or similar
   - Make bucket/files publicly accessible or use signed URLs

2. **Add download script**
   Create `download_models.py`:
   ```python
   import os
   import urllib.request

   MODEL_URLS = {
       'label_mappings.pkl': 'https://your-storage.com/label_mappings.pkl',
       'lemma_dict.pkl': 'https://your-storage.com/lemma_dict.pkl',
       'model_weights.pt': 'https://your-storage.com/model_weights.pt',
   }

   def download_models():
       os.makedirs('final_model7', exist_ok=True)
       for filename, url in MODEL_URLS.items():
           if not os.path.exists(filename):
               print(f"Downloading {filename}...")
               urllib.request.urlretrieve(url, filename)
               print(f"Downloaded {filename}")

   if __name__ == "__main__":
       download_models()
   ```

3. **Update startup in railway.toml**
   ```toml
   [deploy]
   startCommand = "python download_models.py && uvicorn main:app --host 0.0.0.0 --port $PORT"
   ```

## Verification

After deployment, verify model files are loaded:

```bash
# Check health endpoint
curl https://your-app.railway.app/health

# Should show:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "labels_loaded": true,
#   ...
# }
```

## File Size Reference

Check your file sizes:
```bash
du -sh label_mappings.pkl lemma_dict.pkl final_model7/
```

| File | Typical Size | Recommended Method |
|------|-------------|-------------------|
| label_mappings.pkl | < 1 MB | Git (normal) |
| lemma_dict.pkl | 1-50 MB | Git LFS |
| model_weights.pt | > 500 MB | Git LFS or Railway Volume |

## Troubleshooting

### "File not found" errors
- Check file paths in environment variables
- Verify files uploaded successfully
- Check Railway logs: `railway logs`

### "Model not loaded" in health check
- Ensure all required files are present
- Check file permissions
- Verify sufficient memory (upgrade Railway plan if needed)

### Git LFS quota exceeded
- Use Railway volumes instead
- Or use external storage (S3, GCS)

### Slow startup times
- Model loading takes time (30-60s is normal for large models)
- Railway timeout is 100s by default
- Consider caching with persistent volumes

## Best Practices

1. **Version your models**: Use tags or branches for different model versions
2. **Test locally first**: Ensure model loads correctly before deploying
3. **Monitor size**: Keep model sizes reasonable for your hosting plan
4. **Backup files**: Keep copies of model files in secure storage
5. **Document changes**: Note model version in commits/tags

## Need Help?

- Railway Volumes: https://docs.railway.app/reference/volumes
- Git LFS: https://git-lfs.github.com/
- Railway CLI: https://docs.railway.app/develop/cli

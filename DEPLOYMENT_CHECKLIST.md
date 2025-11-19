# 🚀 Deployment Checklist for Railway

Use this checklist to ensure your Karakalpak API is properly deployed to production.

## Pre-Deployment Setup

### ✅ Files Created/Modified
- [x] `main.py` - Updated with security features and production configs
- [x] `requirements.txt` - Pinned dependency versions
- [x] `.gitignore` - Excludes unnecessary files
- [x] `.env.example` - Environment variable template
- [x] `README.md` - Complete documentation
- [x] `Procfile` - Railway start command
- [x] `railway.toml` - Railway configuration
- [x] `runtime.txt` - Python version specification
- [x] `nixpacks.toml` - Build configuration
- [x] `.gitattributes` - Git LFS configuration
- [x] `MODEL_SETUP.md` - Model file management guide

### ✅ Security Improvements
- [x] CORS middleware configured
- [x] Security headers added (X-Content-Type-Options, X-Frame-Options, etc.)
- [x] Input validation (max sentence length, max batch size)
- [x] Request logging middleware
- [x] GZip compression enabled
- [x] API docs can be disabled in production (ENABLE_DOCS=false)

### ✅ Production Features
- [x] Environment variable support for all configurations
- [x] Health check endpoint (`/health`) for Railway monitoring
- [x] Proper error handling and logging
- [x] Graceful startup with missing file checks
- [x] Root endpoint with API information

## Step-by-Step Deployment Guide

### Step 1: Initialize Git Repository

```bash
# Initialize git (if not already done)
git init

# Check what will be committed
git status

# Note: venv/, *.pkl, *.pt files should NOT appear (they're in .gitignore)
```

### Step 2: Handle Model Files

Choose ONE of these methods:

#### Option A: Git LFS (for models < 2GB)
```bash
# Install Git LFS
git lfs install

# Verify tracking (already configured in .gitattributes)
git lfs track "*.pkl" "*.pt" "*.bin"

# Add files
git add .gitattributes
git add label_mappings.pkl lemma_dict.pkl final_model7/

# Verify LFS tracking
git lfs ls-files
# You should see your large files listed
```

#### Option B: Railway Volumes (for models > 2GB)
- Keep model files in .gitignore (already configured)
- Plan to upload them via Railway CLI after deployment
- See MODEL_SETUP.md for detailed instructions

### Step 3: Commit Your Code

```bash
# Add all production-ready files
git add .

# Commit with descriptive message
git commit -m "Initial commit: Production-ready Karakalpak NLP API

- FastAPI backend with multi-task NLP model
- Security: CORS, headers, input validation
- Production configs for Railway deployment
- Comprehensive documentation"

# Verify commit
git log --oneline
```

### Step 4: Create GitHub Repository

1. Go to https://github.com/new
2. Create repository (e.g., "karakalpak-nlp-api")
3. Don't initialize with README (you already have one)

```bash
# Add remote
git remote add origin https://github.com/YOUR_USERNAME/karakalpak-nlp-api.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 5: Deploy to Railway

1. **Go to Railway** (https://railway.app)
2. **Click "New Project"**
3. **Select "Deploy from GitHub repo"**
4. **Authorize GitHub** (if first time)
5. **Select your repository** (karakalpak-nlp-api)
6. **Railway auto-detects** Python and starts building

### Step 6: Configure Environment Variables in Railway

1. In Railway dashboard, click on your service
2. Go to "Variables" tab
3. Add these variables:

```
MODEL_DIR=final_model7
LABELS_PATH=label_mappings.pkl
LEMMA_PATH=lemma_dict.pkl
BACKBONE=xlm-roberta-base
ALLOWED_ORIGINS=*
MAX_SENTENCE_LENGTH=1000
MAX_SENTENCES_PER_REQUEST=50
ENABLE_DOCS=false
PYTHONUNBUFFERED=1
```

**Important for Production:**
- Change `ALLOWED_ORIGINS` from `*` to your actual frontend domain
- Set `ENABLE_DOCS=false` to hide API documentation

### Step 7: Upload Model Files (if using Railway Volumes)

Only if you chose Option B (Railway Volumes):

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Link to your project
railway link

# Create volume
railway volumes create models

# Upload model files
railway volumes upload models label_mappings.pkl
railway volumes upload models lemma_dict.pkl
# ... upload other files
```

### Step 8: Verify Deployment

1. **Check deployment logs** in Railway dashboard
   - Look for "Startup complete" message
   - No errors during model loading

2. **Test health endpoint**
   ```bash
   curl https://your-app-name.railway.app/health
   ```

   Expected response:
   ```json
   {
     "status": "healthy",
     "device": "cpu",
     "model_loaded": true,
     "labels_loaded": true,
     "pos_classes": 17,
     "morph_features": 9,
     "lemma_entries": 50000
   }
   ```

3. **Test API endpoint**
   ```bash
   curl -X POST "https://your-app-name.railway.app/predict" \
     -H "Content-Type: application/json" \
     -d '{"sentence": "Test sentence"}'
   ```

### Step 9: Configure Custom Domain (Optional)

1. In Railway dashboard, go to "Settings"
2. Under "Domains", click "Generate Domain" or add custom domain
3. Follow Railway's instructions for DNS configuration

### Step 10: Monitor and Maintain

- **View logs**: Railway dashboard → Logs tab
- **Monitor metrics**: CPU, Memory, Network usage in dashboard
- **Set up alerts**: Railway can notify you of issues
- **Health checks**: Railway uses `/health` endpoint automatically

## Post-Deployment Checklist

- [ ] Deployment successful (no errors in logs)
- [ ] Health check returns "healthy" status
- [ ] Test API with sample request (works correctly)
- [ ] ALLOWED_ORIGINS set to specific domain (not `*`)
- [ ] ENABLE_DOCS set to `false` (or protected)
- [ ] Custom domain configured (if needed)
- [ ] Model files loaded successfully
- [ ] Response times acceptable
- [ ] Monitor logs for any warnings/errors
- [ ] Document API URL for frontend team
- [ ] Set up monitoring/alerting

## Troubleshooting Common Issues

### Issue: "Model files not found"
**Solution**:
- Verify Git LFS files pushed successfully: `git lfs ls-files`
- Or upload via Railway volumes
- Check environment variable paths

### Issue: "Out of memory"
**Solution**:
- Upgrade Railway plan for more RAM
- Reduce batch size (MAX_SENTENCES_PER_REQUEST)
- Consider using CPU-only PyTorch

### Issue: "Startup timeout"
**Solution**:
- Model loading takes time (up to 60-90s)
- Railway timeout is 100s by default
- Check logs for actual errors

### Issue: "CORS errors in frontend"
**Solution**:
- Update ALLOWED_ORIGINS to include your frontend domain
- Format: `https://frontend.com,https://www.frontend.com`
- Restart Railway service after updating

### Issue: "502 Bad Gateway"
**Solution**:
- Check Railway logs for Python errors
- Verify all dependencies installed correctly
- Ensure PORT environment variable is used

## Additional Resources

- [Railway Documentation](https://docs.railway.app)
- [Git LFS Guide](https://git-lfs.github.com)
- [FastAPI Production Guide](https://fastapi.tiangolo.com/deployment/)
- [MODEL_SETUP.md](./MODEL_SETUP.md) - Model file management
- [README.md](./README.md) - Complete documentation

## Next Steps After Deployment

1. **Frontend Integration**: Share API URL with frontend team
2. **API Documentation**: Share `/docs` endpoint (if enabled) or create API docs
3. **Monitoring**: Set up external monitoring (UptimeRobot, etc.)
4. **Backups**: Ensure model files backed up securely
5. **CI/CD**: Consider GitHub Actions for automated deployments
6. **Security**: Add API key authentication if needed
7. **Rate Limiting**: Consider adding rate limiting for production
8. **Scaling**: Monitor usage and upgrade Railway plan as needed

---

**✅ Deployment Complete!**

Your Karakalpak NLP API is now live on Railway! 🎉

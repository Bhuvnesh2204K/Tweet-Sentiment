# Deploying Twitter Sentiment Analysis to Vercel

## ⚠️ Important Considerations

**Vercel has limitations for ML applications:**
- **Cold starts**: First request may take 10-30 seconds
- **Memory limits**: 1024MB RAM limit
- **Execution time**: 10-30 second timeout
- **File size**: 50MB limit per function

## 🚀 Deployment Steps

### Step 1: Prepare Your Project

1. **Create pre-trained model** (run this locally first):
   ```bash
   python create_model.py
   ```
   This creates `sentiment_model.pkl` and `vectorizer.pkl`

2. **Use the Vercel-optimized files**:
   - `app_vercel.py` (instead of `app.py`)
   - `requirements_vercel.txt` (instead of `requirements.txt`)

### Step 2: Install Vercel CLI

```bash
npm install -g vercel
```

### Step 3: Deploy to Vercel

1. **Login to Vercel**:
   ```bash
   vercel login
   ```

2. **Deploy your project**:
   ```bash
   vercel
   ```

3. **Follow the prompts**:
   - Set up and deploy: `Y`
   - Which scope: Select your account
   - Link to existing project: `N`
   - Project name: `twitter-sentiment-analysis`
   - Directory: `.` (current directory)
   - Override settings: `N`

### Step 4: Configure Vercel

Create a `vercel.json` file in your project root:

```json
{
  "version": 2,
  "builds": [
    {
      "src": "app_vercel.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app_vercel.py"
    }
  ],
  "functions": {
    "app_vercel.py": {
      "maxDuration": 30
    }
  }
}
```

### Step 5: Environment Variables (Optional)

If you need environment variables, add them in the Vercel dashboard or via CLI:

```bash
vercel env add VARIABLE_NAME
```

## 📁 Required Files for Vercel

Make sure these files are in your project:

```
Twitter-Sentiment-Analysis/
├── app_vercel.py           # Main Flask app (Vercel optimized)
├── templates/
│   └── index.html         # Frontend template
├── requirements_vercel.txt # Python dependencies
├── vercel.json            # Vercel configuration
├── sentiment_model.pkl    # Pre-trained model
├── vectorizer.pkl         # Pre-trained vectorizer
├── train_tweet.csv        # Training data
└── test_tweets.csv        # Test data
```

## 🔧 Alternative Deployment Options

If Vercel doesn't work well for your ML app, consider:

### 1. **Railway** (Recommended for ML)
- Better for Python/ML applications
- More generous limits
- Easy deployment

### 2. **Render**
- Good for Python web apps
- Free tier available
- Longer cold start times

### 3. **Heroku**
- Classic choice for Python apps
- Paid service
- Good for ML applications

### 4. **Google Cloud Run**
- Serverless containers
- Good for ML workloads
- More complex setup

## 🐛 Troubleshooting

### Common Issues:

1. **Cold Start Timeouts**:
   - Reduce model complexity
   - Use pre-trained models
   - Increase `maxDuration` in vercel.json

2. **Memory Issues**:
   - Reduce `max_features` in vectorizer
   - Use lighter ML models
   - Optimize data preprocessing

3. **File Size Limits**:
   - Compress model files
   - Use smaller datasets
   - Split large files

### Performance Optimization:

1. **Pre-train models locally** and upload the `.pkl` files
2. **Use TF-IDF instead of CountVectorizer** (smaller memory footprint)
3. **Limit feature count** to 1000 or less
4. **Use simple preprocessing** (no NLTK dependencies)

## 📊 Monitoring

After deployment, monitor:
- Response times
- Error rates
- Memory usage
- Cold start frequency

## 🔄 Updates

To update your deployment:

```bash
vercel --prod
```

## 📞 Support

If you encounter issues:
1. Check Vercel logs in the dashboard
2. Test locally with `vercel dev`
3. Consider alternative platforms for ML applications

## 🎯 Best Practices

1. **Keep models small** (< 50MB total)
2. **Use pre-trained models** when possible
3. **Implement caching** for repeated requests
4. **Monitor performance** regularly
5. **Have a fallback** for cold starts 
# Twitter Sentiment Analysis - Web Frontend

A beautiful web application for analyzing Twitter sentiment using machine learning. This application provides a modern, responsive interface for sentiment analysis with real-time results.

## Features

- 🎨 **Modern UI**: Beautiful, responsive design with gradient backgrounds and smooth animations
- 🤖 **AI-Powered Analysis**: Uses machine learning models to analyze tweet sentiment
- 📊 **Real-time Statistics**: Shows dataset statistics and analysis results
- 📱 **Mobile Responsive**: Works perfectly on desktop, tablet, and mobile devices
- ⚡ **Fast Performance**: Optimized for quick sentiment analysis
- 🎯 **Example Tweets**: Pre-loaded examples to test the system

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the Application

```bash
python app.py
```

The application will start and be available at: **http://localhost:5000**

## How to Use

1. **Open the Application**: Navigate to `http://localhost:5000` in your web browser
2. **Enter a Tweet**: Type or paste a tweet in the text area
3. **Analyze**: Click the "Analyze Sentiment" button
4. **View Results**: See the sentiment prediction and confidence score
5. **Try Examples**: Click on the example tweets to test different sentiments

## Features Explained

### Sentiment Analysis
- **Positive**: Tweets with positive sentiment (label 0)
- **Negative**: Tweets with negative sentiment (label 1)
- **Confidence Score**: Percentage indicating how confident the model is in its prediction

### Dataset Statistics
- **Total Tweets**: Number of tweets in the training dataset
- **Positive Tweets**: Count of positive tweets
- **Negative Tweets**: Count of negative tweets
- **Positive Percentage**: Percentage of positive tweets in the dataset

## Technical Details

### Backend (Flask)
- **Framework**: Flask web framework
- **Model**: Logistic Regression with TF-IDF features
- **Preprocessing**: Text cleaning, stemming, and stopword removal
- **API Endpoints**: 
  - `/` - Main page
  - `/analyze` - Sentiment analysis endpoint
  - `/stats` - Dataset statistics endpoint

### Frontend (HTML/CSS/JavaScript)
- **Design**: Modern gradient design with glassmorphism effects
- **Icons**: Font Awesome icons
- **Fonts**: Inter font family
- **Responsive**: CSS Grid and Flexbox for responsive layout
- **Animations**: Smooth transitions and hover effects

### Machine Learning Pipeline
1. **Text Preprocessing**: Remove special characters, convert to lowercase
2. **Tokenization**: Split text into words
3. **Stemming**: Reduce words to their root form
4. **Stopword Removal**: Remove common words
5. **Feature Extraction**: TF-IDF vectorization
6. **Scaling**: Standardize features
7. **Prediction**: Logistic Regression model

## File Structure

```
Twitter-Sentiment-Analysis/
├── app.py                 # Flask application
├── templates/
│   └── index.html        # Main HTML template
├── requirements.txt      # Python dependencies
├── train_tweet.csv      # Training dataset
├── test_tweets.csv      # Test dataset
├── twitter_sentiment.py # Original analysis script
└── README_FRONTEND.md   # This file
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Change port in app.py or kill existing process
   netstat -ano | findstr :5000
   taskkill /PID <PID> /F
   ```

2. **Missing Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **NLTK Data Missing**
   ```bash
   python -c "import nltk; nltk.download('stopwords')"
   ```

### Performance Tips

- The model loads on startup, so the first analysis might take a few seconds
- Subsequent analyses will be much faster
- The application uses caching for better performance

## Customization

### Changing the Model
You can modify the `load_model()` function in `app.py` to use different machine learning models:

```python
# Example: Using Random Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
```

### Styling
The CSS is embedded in the HTML template. You can modify colors, fonts, and layout by editing the `<style>` section in `templates/index.html`.

## Contributing

Feel free to contribute to this project by:
- Adding new features
- Improving the UI/UX
- Optimizing the machine learning model
- Adding more example tweets
- Implementing additional analysis metrics

## License

This project is open source and available under the MIT License. 
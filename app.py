from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import re
import pickle
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)

# Global variables for the model
model = None
cv = None
sc = None

def preprocess_text(text):
    """Preprocess text for sentiment analysis"""
    # Remove special characters and digits
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    
    # Remove stopwords and apply stemming
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    review = [ps.stem(word) for word in review if not word in stop_words]
    
    # Join back into string
    review = ' '.join(review)
    return review

def load_model():
    """Load the trained model and vectorizer"""
    global model, cv, sc
    
    try:
        # Try to load pre-trained model first
        model = joblib.load('sentiment_model.pkl')
        cv = joblib.load('vectorizer.pkl')
        # Note: The original model doesn't use a scaler, so we'll set it to None
        sc = None
        print("Pre-trained model loaded successfully!")
        return True
    except Exception as e:
        print(f"Pre-trained model not found or error loading: {e}. Creating new model...")
        
        # Load training data to fit vectorizer
        train = pd.read_csv('train_tweet.csv')
        
        # Preprocess training data
        train_corpus = []
        for i in range(len(train)):
            review = preprocess_text(train['tweet'][i])
            train_corpus.append(review)
        
        # Create and fit TfidfVectorizer (matching create_model.py)
        from sklearn.feature_extraction.text import TfidfVectorizer
        cv = TfidfVectorizer(max_features=1000, stop_words='english')
        cv.fit_transform(train_corpus)
        
        # No scaler needed for TfidfVectorizer
        sc = None
        
        # Train a simple model (Logistic Regression) for demo
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        
        # Prepare features
        X = cv.transform(train_corpus)
        y = train['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        # Train model (no scaling needed for TfidfVectorizer)
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        print("Model created and loaded successfully!")
        return True

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    try:
        # Load model if not already loaded
        if model is None:
            load_model()
            
        data = request.get_json()
        tweet = data.get('tweet', '')
        
        if not tweet.strip():
            return jsonify({'error': 'Please enter a tweet to analyze'}), 400
        
        # Preprocess the tweet
        processed_tweet = preprocess_text(tweet)
        
        # Transform using vectorizer
        tweet_vector = cv.transform([processed_tweet])
        
        # Predict sentiment (no scaling needed for TfidfVectorizer)
        prediction = model.predict(tweet_vector)[0]
        probability = model.predict_proba(tweet_vector)[0]
        
        # Get sentiment label and confidence
        sentiment = "Negative" if prediction == 1 else "Positive"
        confidence = probability[1] if prediction == 1 else probability[0]
        
        return jsonify({
            'sentiment': sentiment,
            'confidence': round(confidence * 100, 2),
            'tweet': tweet
        })
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/stats')
def get_stats():
    try:
        # Load data for statistics
        train = pd.read_csv('train_tweet.csv')
        
        # Calculate basic statistics
        total_tweets = len(train)
        positive_tweets = len(train[train['label'] == 0])
        negative_tweets = len(train[train['label'] == 1])
        
        # Calculate percentages
        positive_percent = round((positive_tweets / total_tweets) * 100, 1)
        negative_percent = round((negative_tweets / total_tweets) * 100, 1)
        
        return jsonify({
            'total_tweets': total_tweets,
            'positive_tweets': positive_tweets,
            'negative_tweets': negative_tweets,
            'positive_percent': positive_percent,
            'negative_percent': negative_percent
        })
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

# For local development
if __name__ == '__main__':
    import os
    print("Loading model...")
    load_model()
    print("Starting Flask app...")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port) 
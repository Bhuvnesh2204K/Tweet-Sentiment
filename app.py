from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import re
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
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
    
    # Load training data to fit vectorizer
    train = pd.read_csv('train_tweet.csv')
    
    # Preprocess training data
    train_corpus = []
    for i in range(len(train)):
        review = preprocess_text(train['tweet'][i])
        train_corpus.append(review)
    
    # Create and fit CountVectorizer
    cv = CountVectorizer(max_features=2500)
    cv.fit_transform(train_corpus)
    
    # Create StandardScaler
    sc = StandardScaler()
    
    # Train a simple model (Logistic Regression) for demo
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    
    # Prepare features
    X = cv.transform(train_corpus).toarray()
    y = train['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Scale features
    X_train_scaled = sc.fit_transform(X_train)
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    print("Model loaded successfully!")

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
        
        # Transform using CountVectorizer
        tweet_vector = cv.transform([processed_tweet]).toarray()
        
        # Scale the features
        tweet_scaled = sc.transform(tweet_vector)
        
        # Predict sentiment
        prediction = model.predict(tweet_scaled)[0]
        probability = model.predict_proba(tweet_scaled)[0]
        
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
    print("Loading model...")
    load_model()
    print("Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5000) 
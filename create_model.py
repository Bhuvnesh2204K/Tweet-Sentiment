import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

def preprocess_text(text):
    """Simple text preprocessing"""
    # Remove special characters and digits
    text = re.sub('[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text

def create_and_save_model():
    """Create and save the sentiment analysis model"""
    print("Loading training data...")
    train = pd.read_csv('train_tweet.csv')
    
    print("Preprocessing tweets...")
    train['processed_tweet'] = train['tweet'].apply(preprocess_text)
    
    print("Creating TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(train['processed_tweet'])
    y = train['label']
    
    print("Training Logistic Regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    
    print("Saving model and vectorizer...")
    joblib.dump(model, 'sentiment_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    
    print("Model created and saved successfully!")
    
    # Test the model
    test_tweet = "I love this amazing product!"
    processed_test = preprocess_text(test_tweet)
    test_vector = vectorizer.transform([processed_test])
    prediction = model.predict(test_vector)[0]
    probability = model.predict_proba(test_vector)[0]
    
    sentiment = "Negative" if prediction == 1 else "Positive"
    confidence = probability[1] if prediction == 1 else probability[0]
    
    print(f"Test tweet: '{test_tweet}'")
    print(f"Prediction: {sentiment} (Confidence: {confidence:.2%})")

if __name__ == "__main__":
    create_and_save_model() 
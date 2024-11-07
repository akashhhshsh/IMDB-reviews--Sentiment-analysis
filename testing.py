import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load the saved model, vectorizer, and label encoder
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

def remove_html(text):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

def remove_urls(text):
    url_pattern = r'http[s]?://\S+'
    return re.sub(url_pattern, '', text)

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

def preprocess_review(review):
    review = review.lower()
    review = remove_html(review)
    review = remove_urls(review)
    review = remove_punctuation(review)
    tokens = word_tokenize(review)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)

# Function to predict sentiment of a custom review
def predict_sentiment(review):
    processed_review = preprocess_review(review)
    review_tfidf = vectorizer.transform([processed_review])
    prediction = model.predict(review_tfidf)
    
    # Decode the prediction to 'Positive' or 'Negative'
    sentiment = label_encoder.inverse_transform(prediction)[0]
    
    return sentiment

# Get custom review from the user
user_review = input("Enter your review: ")

# Predict sentiment
predicted_sentiment = predict_sentiment(user_review)

print(f"The sentiment of the review is: {predicted_sentiment}")

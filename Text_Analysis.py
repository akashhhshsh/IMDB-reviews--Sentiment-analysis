import pandas as pd

# Loading the dataset
data = pd.read_csv('/Users/akashadhyapak/Documents/ML/Text Processing/IMDB Dataset.csv')

# Converting the review column to lower case
data['review'] = data['review'].str.lower()

# Removing HTML tags
from bs4 import BeautifulSoup

def remove_html(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

data['review'] = data['review'].apply(remove_html)

# Removing URLs
import re

def remove_urls(text):
    url_pattern = r'http[s]?://\S+'
    return re.sub(url_pattern, '', text)

data['review'] = data['review'].apply(remove_urls)

# Removing Punctuations
def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

data['review'] = data['review'].apply(remove_punctuation)

#Removing Spelling Errors  (Commented because it takes too too toooo long to run, Goodluck running it :)
# from textblob import TextBlob

# def correct_spelling(text):
#     return str(TextBlob(text).correct())

# data['review'] = data['review'].apply(correct_spelling)  
# print(data.head())

# Tokenizing words
import nltk
#nltk.download('punkt')

from nltk.tokenize import word_tokenize

def tokenize_text(text):
    return word_tokenize(text)

data['review'] = data['review'].apply(tokenize_text)

# Removing Stopwords
from nltk.corpus import stopwords
#nltk.download('stopwords')

stopwords = set(stopwords.words('english'))

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stopwords]

data['review'] = data['review'].apply(remove_stopwords)

# Stemming
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

def stem_words(tokens):
    return [stemmer.stem(word) for word in tokens]

data['review'] = data['review'].apply(stem_words)

# Sentiment Analysis

# Encoding the data
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

data['sentiment'] = label_encoder.fit_transform(data['sentiment'])

# Split the data
from sklearn.model_selection import train_test_split

X = data['review']  
y = data['sentiment'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the split
print("Training data size:", len(X_train))
print("Test data size:", len(X_test))

# Convert tokenized text back to single strings
X_train = X_train.apply(lambda x: ' '.join(x))  # Join words into a single string
X_test = X_test.apply(lambda x: ' '.join(x))  # Join words into a single string

# Apply TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)  # Limit to the top 5000 features (words)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training data shape:", X_train_tfidf.shape)
print("Test data shape:", X_test_tfidf.shape)

# Train the model using Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

model = LogisticRegression()

# Train the model
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title("Confusion Matrix")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

import joblib

# Save the trained model to a file
joblib.dump(model, 'sentiment_model.pkl')

# Save the TF-IDF vectorizer to a file
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Optionally, save the label encoder (if you want to decode predictions in the future)
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Model, vectorizer, and label encoder saved successfully!")


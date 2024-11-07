IMDB Movie Review Sentiment Analysis
This project implements a sentiment analysis model that classifies movie reviews from the IMDB dataset into Positive or Negative sentiments. The model uses Natural Language Processing (NLP) techniques such as text preprocessing, tokenization, stopword removal, stemming, and TF-IDF vectorization. A Logistic Regression model is used for classification.

Link for the dataset --->  https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data

Project Overview
Dataset: IMDB Movie Reviews
Model: Logistic Regression
Text Processing: NLTK, BeautifulSoup, Regex
Libraries Used: Pandas, scikit-learn, NLTK, joblib
Prediction: Sentiment classification of movie reviews as Positive or Negative.
Steps Involved
Data Preprocessing:

Loading the dataset
Converting text to lowercase
Removing HTML tags and URLs
Removing punctuation
Tokenizing the reviews
Removing stopwords
Stemming the words
Model Training:

TF-IDF vectorization to convert text data into numerical features
Logistic Regression model training using the preprocessed data
Sentiment Prediction:

User can input a review, and the model will predict if it's Positive or Negative.
Files in the Project
imdb.py: Contains the main script for training the model and preprocessing the data.
testing.py: Separate script for loading the trained model and making predictions on new reviews.
sentiment_model.pkl: The saved Logistic Regression model.
tfidf_vectorizer.pkl: The saved TF-IDF vectorizer.
label_encoder.pkl: The saved Label Encoder.


How to Use
Install Dependencies: Install the necessary libraries using pip

Training the Model: Run the imdb.py script to preprocess the data, train the model, and save the model.
python imdb.py


Making Predictions: Run the testing.py script to input a review and get the sentiment prediction.
python testing.py


Customization: You can modify the dataset or experiment with different models for better accuracy.

Conclusion
This project demonstrates the application of NLP techniques to classify text data and build a machine learning model for sentiment analysis. It's a great starting point for anyone looking to dive into text-based classification tasks.

Results-
<img width="603" alt="Screenshot 2024-11-07 at 5 25 17 PM" src="https://github.com/user-attachments/assets/e561946a-f94d-4980-925f-c923ded85398">
<img width="780" alt="Screenshot 2024-11-07 at 5 25 38 PM" src="https://github.com/user-attachments/assets/890e7325-edc0-41d7-8fad-2d6f963ecff6">
<img width="474" alt="Screenshot 2024-11-07 at 5 32 10 PM" src="https://github.com/user-attachments/assets/42195f9e-ce5b-49dd-83c5-ca8c76d2ae6d">
<img width="487" alt="Screenshot 2024-11-07 at 5 32 32 PM" src="https://github.com/user-attachments/assets/f9d9f6d7-5dbc-4a2a-aebc-75d50edbe26d">
<img width="450" alt="Screenshot 2024-11-07 at 5 32 49 PM" src="https://github.com/user-attachments/assets/b4946658-20b0-4af0-ab1c-362112ac5fb9">
<img width="866" alt="Screenshot 2024-11-07 at 5 33 22 PM" src="https://github.com/user-attachments/assets/3b9cc048-a544-49d3-92b5-07579d978b57">



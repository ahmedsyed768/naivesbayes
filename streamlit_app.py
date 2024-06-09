import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import io
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline

# Downloading necessary NLTK data
nltk.download('vader_lexicon')

class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.clf = None

    def preprocess_text(self, text):
        if isinstance(text, str):
            return text.lower()
        else:
            return str(text).lower()

    def train_classifier(self, reviews, labels):
        vectorizer = CountVectorizer(preprocessor=self.preprocess_text)
        X = vectorizer.fit_transform(reviews)
        
        if X.shape[0] != len(labels):
            raise ValueError("Number of samples in features and labels must be the same.")
        
        self.clf = MultinomialNB()
        self.clf.fit(X, labels)
        return make_pipeline(vectorizer, self.clf)

    def analyze_sentiment(self, review):
        sentiment_score = self.sia.polarity_scores(str(review))["compound"]
        return sentiment_score

# Update Streamlit UI setup
st.title("Student Review Sentiment Analysis for Teaching")

# Upload CSV file
csv_file = st.file_uploader("Upload your CSV file")

if csv_file:
    df = pd.read_csv(io.BytesIO(csv_file.read()), encoding='utf-8')
    st.write(df.head())  # Debug statement to check the loaded data

    # Perform sentiment analysis
    analyzer = SentimentAnalyzer()

    # Focus on the "teaching" column
    if 'teaching' in df.columns:
        teaching_reviews = df['teaching'].dropna().astype(str).tolist()
        teaching_sentiments = [analyzer.analyze_sentiment(review) for review in teaching_reviews]

        # Calculate overall sentiment score
        overall_teaching_sentiment = sum(teaching_sentiments) / len(teaching_sentiments)

        # Plotting sentiment analysis for the "teaching" category
        fig, ax = plt.subplots()
        categories = ['Teaching']
        sentiment_scores = [overall_teaching_sentiment]

        ax.bar(categories, sentiment_scores, color='blue')
        ax.set_xlabel('Feedback Category')
        ax.set_ylabel('Sentiment Score')
        ax.set_title('Overall Sentiment Analysis for Teaching')
        st.pyplot(fig)

        # Displaying descriptions
        st.subheader("Overall Sentiment Description for Teaching")
        if overall_teaching_sentiment >= 0.65:
            description = "Excellent progress, keep up the good work!"
        elif overall_teaching_sentiment >= 0.62:
            description = "Good progress, continue to work hard!"
        else:
            description = "Needs improvement, stay motivated and keep trying!"
        st.write(f"**Teaching**: {description}")

        # Detailed breakdown of sentiments
        st.subheader("Detailed Breakdown of Sentiments for Teaching")
        sentiment_breakdown = pd.DataFrame(teaching_sentiments, columns=['Sentiment Score'])
        st.write(sentiment_breakdown)

        # Train Naive Bayes classifier
        st.subheader("Naive Bayes Classifier")
        labels = [1 if sentiment >= 0.65 else 0 for sentiment in teaching_sentiments]
        pipeline = analyzer.train_classifier(teaching_reviews, labels)
        st.write("Classifier trained successfully.")

        # Prediction on new data
        test_reviews = st.text_area("Enter reviews for prediction (separate each review with a new line):")
        if test_reviews:
            test_reviews_list = test_reviews.split('\n')
            predictions = pipeline.predict(test_reviews_list)
            st.write("Predictions:")
            st.write(predictions)
    else:
        st.write("The dataset does not contain the 'teaching' column.")

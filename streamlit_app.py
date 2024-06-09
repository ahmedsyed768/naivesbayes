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

    def predict_sentiments(self, test_reviews):
        return self.clf.predict(test_reviews)

    def transform_scale(self, score):
        return 5 * score + 5  # Convert the sentiment score from -1 to 1 scale to 0 to 10 scale

    def calculate_overall_sentiment(self, reviews):
        compound_scores = [self.sia.polarity_scores(str(review))["compound"] for review in reviews if isinstance(review, str)]
        overall_sentiment = sum(compound_scores) / len(compound_scores) if compound_scores else 0
        return self.transform_scale(overall_sentiment)

    def analyze_sentiment(self, reviews):
        sentiments = [{'compound': self.transform_scale(self.sia.polarity_scores(str(review))["compound"]),
                       'pos': self.sia.polarity_scores(str(review))["pos"],
                       'neu': self.sia.polarity_scores(str(review))["neu"],
                       'neg': self.sia.polarity_scores(str(review))["neg"]}
                      for review in reviews if isinstance(review, str)]
        return sentiments

    def interpret_sentiment(self, sentiments):
        avg_sentiment = sum([sentiment['compound'] for sentiment in sentiments]) / len(sentiments) if sentiments else 0
        if avg_sentiment >= 6.5:
            description = "Excellent progress, keep up the good work!"
        elif avg_sentiment >= 6.2:
            description = "Good progress, continue to work hard!"
        else:
            description = "Needs improvement, stay motivated and keep trying!"

        return description

# Update Streamlit UI setup
st.title("Student Review Sentiment Analysis for Teaching")

# Upload CSV file
csv_file = st.file_uploader("Upload your CSV file")

if csv_file:
    df = pd.read_csv(io.BytesIO(csv_file.read()), encoding='utf-8')
    st.write(df.head())  # Debug statement to check the loaded data

    # Perform sentiment analysis on the "teaching" column
    analyzer = SentimentAnalyzer()

    if 'teaching' in df.columns:
        teaching_reviews = df['teaching'].dropna().astype(str).tolist()
        teaching_sentiments = analyzer.analyze_sentiment(teaching_reviews)
        overall_teaching_sentiment = analyzer.calculate_overall_sentiment(teaching_reviews)

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
        sentiment_description = analyzer.interpret_sentiment(teaching_sentiments)
        st.write(f"**Teaching**: {sentiment_description}")

        # Detailed breakdown of sentiments
        st.subheader("Detailed Breakdown of Sentiments for Teaching")
        sentiment_breakdown = pd.DataFrame(teaching_sentiments)
        st.write(sentiment_breakdown)

        # Train Naive Bayes classifier
        st.subheader("Naive Bayes Classifier")
        labels = [1 if sentiment['compound'] >= 6.5 else 0 for sentiment in teaching_sentiments]
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

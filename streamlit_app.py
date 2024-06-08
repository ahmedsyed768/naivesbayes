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

class SentimentAnalyzer1:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

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

        trend = "No change"
        if len(sentiments) > 1:
            first_half_avg = sum([sentiment['compound'] for sentiment in sentiments[:len(sentiments)//2]]) / (len(sentiments)//2)
            second_half_avg = sum([sentiment['compound'] for sentiment in sentiments[len(sentiments)//2:]]) / (len(sentiments)//2)
            if second_half_avg > first_half_avg:
                trend = "Improving"
            elif second_half_avg < first_half_avg:
                trend = "Declining"

        return description, trend

class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.clf = None

    def train_classifier(self, reviews, labels):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(reviews)
        self.clf = MultinomialNB()
        self.clf.fit(X, labels)
        return make_pipeline(vectorizer, self.clf)

    def predict_sentiments(self, test_reviews):
        return self.clf.predict(test_reviews)

    def transform_scale(self, score):
        return 5 * score + 5  # Convert the sentiment score from -1 to 1 scale to 0 to 10 scale

    def analyze_sentiment(self, reviews):
        sentiments = [{'compound': self.transform_scale(self.sia.polarity_scores(str(review))["compound"]),
                       'pos': self.sia.polarity_scores(str(review))["pos"],
                       'neu': self.sia.polarity_scores(str(review))["neu"],
                       'neg': self.sia.polarity_scores(str(review))["neg"]}
                      for review in reviews if isinstance(review, str)]
        return sentiments

    def calculate_overall_sentiment(self, reviews):
        compound_scores = [self.sia.polarity_scores(str(review))["compound"] for review in reviews if isinstance(review, str)]
        overall_sentiment = sum(compound_scores) / len(compound_scores) if compound_scores else 0
        return self.transform_scale(overall_sentiment)

    def interpret_sentiment(self, sentiments):
        avg_sentiment = sum([sentiment['compound'] for sentiment in sentiments]) / len(sentiments) if sentiments else 0
        if avg_sentiment >= 6.5:
            description = "Excellent progress, keep up the good work!"
        elif avg_sentiment >= 6.2:
            description = "Good progress, continue to work hard!"
        else:
            description = "Needs improvement, stay motivated and keep trying!"

        trend = "No change"
        if len(sentiments) > 1:
            first_half_avg = sum([sentiment['compound'] for sentiment in sentiments[:len(sentiments)//2]]) / (len(sentiments)//2)
            second_half_avg = sum([sentiment['compound'] for sentiment in sentiments[len(sentiments)//2:]]) / (len(sentiments)//2)
            if second_half_avg > first_half_avg:
                trend = "Improving"
            elif second_half_avg < first_half_avg:
                trend = "Declining"

        return description, trend

# Update Streamlit UI setup
st.title("Student Review Sentiment Analysis")

# Upload CSV file
csv_file = st.file_uploader("Upload your CSV file")

if csv_file:
    df = pd.read_csv(io.BytesIO(csv_file.read()), encoding='utf-8')
    st.write(df.head())  # Debug statement to check the loaded data

    # Perform sentiment analysis
    analyzer = SentimentAnalyzer()

    # Columns to analyze
    feedback_columns = ['teaching', 'library_facilities', 'examination', 'labwork', 'extracurricular', 'coursecontent']
    sentiments = {}

    for column in feedback_columns:
        if column in df.columns:
            reviews = df[column].dropna().astype(str).tolist()
            sentiments[column] = analyzer.analyze_sentiment(reviews)

    overall_sentiments = {column: analyzer.calculate_overall_sentiment(df[column].dropna().astype(str).tolist()) for column in feedback_columns}

    # Plotting sentiment analysis for all categories
    fig, ax = plt.subplots()
    categories = list(overall_sentiments.keys())
    sentiment_scores = list(overall_sentiments.values())

    ax.bar(categories, sentiment_scores, color=['blue', 'green', 'gray', 'red', 'purple', 'orange'])
    ax.set_xlabel('Feedback Categories')
    ax.set_ylabel('Sentiment Score')
    ax.set_title('Overall Sentiment Analysis for Feedback Categories')
    st.pyplot(fig)

    # Displaying descriptions
    st.subheader("Overall Sentiment Descriptions")
    for column in feedback_columns:
        sentiment_description, _ = analyzer.interpret_sentiment(sentiments[column])
        st.write(f"**{column.capitalize()}**: {sentiment_description}")

    # Train Naive Bayes classifier
    st.subheader("Naive Bayes Classifier")
    reviews = df[feedback_columns].values.flatten().tolist()
    labels = [1 if sentiment['compound'] >= 6.5 else 0 for sublist in sentiments.values() for sentiment in sublist]
    pipeline = analyzer.train_classifier(reviews, labels)
    st.write("Classifier trained successfully.")

    # Prediction on new data
    test_reviews = st.text_area("Enter reviews for prediction (separate each review with a new line):")
    if test_reviews:
        test_reviews_list = test_reviews.split('\n')
        predictions = pipeline.predict(test_reviews_list)
        st.write("Predictions:")
        st.write(predictions)

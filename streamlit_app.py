import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
import streamlit as st

# Download necessary NLTK data
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
        return self.sia.polarity_scores(str(review))

# Streamlit UI setup
st.title("Student Review Sentiment Analysis")

# Load the dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='utf-8')
    st.write(df.head())  # Display the first few rows of the dataset

    # Perform sentiment analysis
    analyzer = SentimentAnalyzer()

    # Columns to analyze
    feedback_columns = ['teaching']
    sentiments = {column: [] for column in feedback_columns}

    for column in feedback_columns:
        if column in df.columns:
            for review in df[column]:
                sentiment = analyzer.analyze_sentiment(review)
                sentiments[column].append(sentiment)

    # Plotting sentiment analysis for all categories
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = {'positive': 'green', 'neutral': 'blue', 'negative': 'red', 'overall': 'gray'}

    for column in feedback_columns:
        if column in sentiments:
            scores = [s['compound'] for s in sentiments[column]]
            pos_scores = [s['pos'] for s in sentiments[column]]
            neu_scores = [s['neu'] * 0.5 for s in sentiments[column]]  # Scale down neutral scores
            neg_scores = [s['neg'] for s in sentiments[column]]

            ax.plot(scores, label=f"{column.capitalize()} - Overall", color=colors['overall'], linewidth=2)
            ax.plot(pos_scores, label=f"{column.capitalize()} - Positive", color=colors['positive'], linestyle='dotted')
            ax.plot(neu_scores, label=f"{column.capitalize()} - Neutral", color=colors['neutral'], linestyle='dashed', alpha=0.5)
            ax.plot(neg_scores, label=f"{column.capitalize()} - Negative", color=colors['negative'], linestyle='dotted')

    ax.set_xlabel('Review Index')
    ax.set_ylabel('Sentiment Score')
    ax.set_title('Sentiment Analysis')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    st.pyplot(fig)

    # Displaying overall sentiment descriptions
    st.subheader("Overall Sentiment Descriptions")
    for column in feedback_columns:
        avg_sentiment = sum([s['compound'] for s in sentiments[column]]) / len(sentiments[column])
        if avg_sentiment >= 0.65:
            description = "Excellent progress, keep up the good work!"
        elif avg_sentiment >= 0.62:
            description = "Good progress, continue to work hard!"
        else:
            description = "Needs improvement, stay motivated and keep trying!"
        st.write(f"**{column.capitalize()}**: {description}")

    # Train Naive Bayes classifier
    st.subheader("Naive Bayes Classifier")
    reviews = df[feedback_columns].values.flatten().tolist()
    labels = [1 if s['compound'] >= 0.65 else 0 for column in feedback_columns for s in sentiments[column]]
    pipeline = analyzer.train_classifier(reviews, labels)
    st.write("Classifier trained successfully.")

    # Prediction on new data
    test_reviews = st.text_area("Enter reviews for prediction (separate each review with a new line):")
    if test_reviews:
        test_reviews_list = test_reviews.split('\n')
        predictions = pipeline.predict(test_reviews_list)
        st.write("Predictions:")
        st.write(predictions)
else:
    st.write("Please upload a CSV file to proceed.")

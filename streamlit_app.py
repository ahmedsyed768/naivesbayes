import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import io

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

    def transform_scale(self, score):
        return 5 * score + 5  # Convert the sentiment score from -1 to 1 scale to 0 to 10 scale

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

# Update Streamlit UI setup
st.title("Student Review Sentiment Analysis")

# Upload CSV file
csv_file = st.file_uploader("Upload your CSV file")

if csv_file:
    df = pd.read_csv(io.BytesIO(csv_file.read()), encoding='utf-8')
    st.write(df.head())  # Debug statement to check the loaded data

    # Perform sentiment analysis
    analyzer = SentimentAnalyzer()

    # Ensure the column names are case insensitive and match the dataset
    required_columns = ['Teaching', 'CourseContent', 'Examination', 'LabWork', 'Library_Facilities', 'ExtraCurricular']
    lower_columns = {col.lower(): col for col in df.columns}
    review_columns = [lower_columns[col.lower()] for col in required_columns if col.lower() in lower_columns]

    if len(review_columns) != len(required_columns):
        missing_columns = set(required_columns) - set([col for col in lower_columns.values() if col.lower() in [col.lower() for col in review_columns]])
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
    else:
        reviews = df[review_columns].values.flatten().tolist()

        analyzer = SentimentAnalyzer1()

        review_period = st.selectbox("Review Period:", [1, 4])

        if review_period == 1:
            sentiments = analyzer.analyze_sentiment(reviews)
        else:
            sentiments = analyzer.analyze_periodic_sentiment(reviews, review_period)

        overall_sentiment = analyzer.calculate_overall_sentiment(reviews)
        st.subheader(f"Overall Sentiment: {overall_sentiment:.2f}")
        st.subheader("Sentiment Analysis")

        # Plotting sentiment
        weeks = list(range(1, len(sentiments) + 1))
        sentiment_scores = [sentiment['compound'] for sentiment in sentiments]
        pos_scores = [sentiment['pos'] for sentiment in sentiments]
        neu_scores = [sentiment['neu'] for sentiment in sentiments]
        neg_scores = [sentiment['neg'] for sentiment in sentiments]

        fig, ax = plt.subplots()
        ax.plot(weeks, sentiment_scores, label="Overall", color="blue")
        ax.fill_between(weeks, sentiment_scores, color="blue", alpha=0.1)
        ax.plot(weeks, pos_scores, label="Positive", color="green")
        ax.plot(weeks, neu_scores, label="Neutral", color="gray")
        ax.plot(weeks, neg_scores, label="Negative", color="red")

        ax.set_xlabel('Week')
        ax.set_ylabel('Sentiment Score')
        ax.set_title('Sentiment Analysis')
        ax.legend()
        st.pyplot(fig)

        description, trend = analyzer.interpret_sentiment(sentiments)
        st.subheader("Progress Description")
        st.write(f"Sentiment Trend: {trend}")
        st.write(f"Description: {description}")

        # Breakdown of analysis
        st.subheader("Breakdown of Analysis")
        breakdown_df = pd.DataFrame(sentiments, index=list(range(1, len(sentiments) + 1)))
        st.write(breakdown_df)

    else:
        st.write("The uploaded CSV does not match the expected structure.")

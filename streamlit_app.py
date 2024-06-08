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

    def analyze_periodic_sentiment(self, reviews, period):
        period_reviews = [' '.join(reviews[i:i + period]) for i in range(0, len(reviews), period)]
        return self.analyze_sentiment(period_reviews)

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
    if all(col.lower() in map(str.lower, df.columns) for col in required_columns):
        # Use the exact column names in your dataset
        review_columns = ['Teaching', 'CourseContent', 'Examination', 'LabWork', 'Library_Facilities', 'ExtraCurricular']
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

    elif len(df.columns) >= 7 and df.columns[0].lower() == 'student' and all(col.lower().startswith('week') for col in df.columns[1:]):
        # Data structure suggests weekly sentiment analysis
        # Initialize lists to store sentiment scores and labels
        all_reviews = []
        sentiment_labels = []

        # Analyze sentiment for each week
        weekly_sentiments = {}
        for column in df.columns[1:]:
            weekly_reviews = df[column].dropna().astype(str).tolist()
            all_reviews.extend(weekly_reviews)
            analyzed_sentiments = analyzer.analyze_sentiment(weekly_reviews)

            # Store weekly sentiments for visualization
            weekly_sentiments[column] = analyzed_sentiments

            # Extract compound scores and determine sentiment labels (binary classification)
            compound_scores = [sentiment['compound'] for sentiment in analyzed_sentiments]
            weekly_labels = [1 if score > 5 else 0 for score in compound_scores]
            sentiment_labels.extend(weekly_labels)

        # Plotting sentiment trends
        weeks = list(range(1, len(df.columns)))
        sentiment_scores = [sum([sentiment['compound'] for sentiment in weekly_sentiments[column]]) / len(weekly_sentiments[column]) for column in df.columns[1:]]
        pos_scores = [sum([sentiment['pos'] for sentiment in weekly_sentiments[column]]) / len(weekly_sentiments[column]) for column in df.columns[1:]]
        neu_scores = [sum([sentiment['neu'] for sentiment in weekly_sentiments[column]]) / len(weekly_sentiments[column]) for column in df.columns[1:]]
        neg_scores = [sum([sentiment['neg'] for sentiment in weekly_sentiments[column]]) / len(weekly_sentiments[column]) for column in df.columns[1:]]

        fig, ax = plt.subplots()
        ax.plot(weeks, sentiment_scores, label="Overall", color="blue")
        ax.fill_between(weeks, sentiment_scores, color="blue", alpha=0.1)
        ax.plot(weeks, pos_scores, label="Positive", color="green")
        ax.plot(weeks, neu_scores, label="Neutral", color="gray")
        ax.plot(weeks, neg_scores, label="Negative", color="red")

        ax.set_xlabel('Week')
        ax.set_ylabel('Sentiment Score')
        ax.set_title('Weekly Sentiment Analysis')
        ax.legend()
        st.pyplot(fig)

        # Train and evaluate a Naive Bayes classifier for sentiment classification
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(all_reviews)
        y = sentiment_labels

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = MultinomialNB()
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        st.subheader("Naive Bayes Classifier Results")
        st.write(f"Accuracy: {accuracy:.2f}")

        # Display sentiment trends and classifier results
        st.subheader("Weekly Sentiment Trends")
        st.write(f"Accuracy of sentiment classifier: {accuracy:.2f}")

        # Breakdown of analysis
        st.subheader("Breakdown of Analysis")
        breakdown_data = {column: weekly_sentiments[column] for column in df.columns[1:]}
        for week, sentiments in breakdown_data.items():
            st.write(f"Week {week}")
            breakdown_df = pd.DataFrame(sentiments)
            st.write(breakdown_df)

    else:
        st.write("The uploaded CSV does not match the expected structure.")

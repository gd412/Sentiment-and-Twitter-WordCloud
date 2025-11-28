import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

st.set_page_config(page_title="Tweet Analytics Dashboard", layout="wide")

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Sentiment Analytics",
        "WordCloud",
        "Hashtags",
        "Influential Tweets",
        "Search Tweets",
        "ML Model",          # <<< ADDED
        "Live Prediction"     # <<< ADDED
    ]
)

# ---------------------------
# Upload CSV
# ---------------------------
uploaded_file = st.sidebar.file_uploader("Upload your tweets CSV", type=["csv"])
df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Helper: Extract hashtags
def extract_hashtags(text):
    return re.findall(r"#(\w+)", str(text))


# ----------------------------------------------------------
# Overview Page
# ----------------------------------------------------------
if page == "Overview":
    st.title("📊 Overview Dashboard")

    if df is None:
        st.warning("Upload a CSV file to continue.")
    else:
        if "label" not in df.columns:
            st.error("CSV must contain a 'label' column.")
        else:
            st.subheader("📅 Filter by Date")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", df["timestamp"].min())
            with col2:
                end_date = st.date_input("End Date", df["timestamp"].max())

            filtered = df[(df["timestamp"] >= pd.to_datetime(start_date)) &
                          (df["timestamp"] <= pd.to_datetime(end_date))]

            st.success(f"Showing {len(filtered)} tweets")

            # KPIs
            st.subheader("📌 Key Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Tweets", len(filtered))
            col2.metric("Positive", (filtered["label"] == "positive").sum())
            col3.metric("Negative", (filtered["label"] == "negative").sum())
            col4.metric("Neutral", (filtered["label"] == "neutral").sum())
            col5.metric("Avg Likes", round(filtered["likes"].mean(), 2))

            # Sentiment distribution
            st.subheader("📈 Sentiment Distribution")
            counts = filtered["label"].value_counts()
            fig, ax = plt.subplots()
            ax.bar(counts.index, counts.values)
            st.pyplot(fig)


# ----------------------------------------------------------
# Sentiment Analytics
# ----------------------------------------------------------
elif page == "Sentiment Analytics":
    st.title("📊 Sentiment Analytics")

    if df is None:
        st.warning("Upload CSV to continue.")
    else:
        sentiment = st.selectbox("Select sentiment", ["positive", "negative", "neutral"])
        filtered = df[df["label"] == sentiment]

        st.dataframe(filtered[["tweet", "likes", "retweets", "device", "timestamp"]])

        st.subheader("Likes vs Retweets")
        fig, ax = plt.subplots()
        ax.scatter(filtered["likes"], filtered["retweets"])
        st.pyplot(fig)


# ----------------------------------------------------------
# WordCloud Page
# ----------------------------------------------------------
elif page == "WordCloud":
    st.title("☁ WordCloud Generator")

    if df is None:
        st.warning("Upload CSV")
    else:
        mode = st.selectbox(
            "Choose Mode",
            ["All Tweets", "By Sentiment", "By Device", "By Location"]
        )

        if mode == "All Tweets":
            text = " ".join(df["tweet"].astype(str))
        elif mode == "By Sentiment":
            sent = st.selectbox("Select Sentiment", ["positive", "negative", "neutral"])
            text = " ".join(df[df["label"] == sent]["tweet"])
        elif mode == "By Device":
            dev = st.selectbox("Select Device", df["device"].unique())
            text = " ".join(df[df["device"] == dev]["tweet"])
        else:
            loc = st.selectbox("Select Location", df["location"].unique())
            text = " ".join(df[df["location"] == loc]["tweet"])

        wc = WordCloud(width=1200, height=600, background_color="black").generate(text)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)


# ----------------------------------------------------------
# Hashtag Analytics
# ----------------------------------------------------------
elif page == "Hashtags":
    st.title("🏷 Top Hashtags")

    if df is None:
        st.warning("Upload CSV")
    else:
        df["hashtags"] = df["tweet"].apply(extract_hashtags)
        all_tags = sum(df["hashtags"], [])

        if not all_tags:
            st.warning("No hashtags found.")
        else:
            tag_series = pd.Series(all_tags).value_counts().head(20)
            st.bar_chart(tag_series)


# ----------------------------------------------------------
# Influential Tweets
# ----------------------------------------------------------
elif page == "Influential Tweets":
    st.title("🔥 Most Influential Tweets")

    if df is None:
        st.warning("Upload CSV")
    else:
        df["engagement"] = df["likes"] + df["retweets"]
        top = df.sort_values(by="engagement", ascending=False).head(20)
        st.dataframe(top[["tweet", "label", "likes", "retweets", "engagement"]])


# ----------------------------------------------------------
# Search Tweets
# ----------------------------------------------------------
elif page == "Search Tweets":
    st.title("🔍 Search Tweets")

    if df is None:
        st.warning("Upload CSV")
    else:
        query = st.text_input("Enter keyword to search:")
        if query:
            result = df[df["tweet"].str.contains(query, case=False, na=False)]
            st.dataframe(result)


# ----------------------------------------------------------
# 🔥 ML Sentiment Model (Training + Switchable Models)
# ----------------------------------------------------------
elif page == "ML Model":
    st.title("🤖 Train Machine Learning Sentiment Model")

    if df is None:
        st.warning("Upload CSV to train ML model.")
    else:
        model_choice = st.selectbox(
            "Select ML Algorithm",
            ["Logistic Regression", "Naive Bayes", "SVM"]
        )

        if st.button("Train Model"):
            X = df["tweet"].astype(str)
            y = df["label"]

            vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
            X_vec = vectorizer.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_vec, y, test_size=0.2, random_state=42
            )

            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=500)
            elif model_choice == "Naive Bayes":
                model = MultinomialNB()
            else:
                model = SVC(kernel="linear")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            st.success(f"{model_choice} trained! Accuracy: {acc*100:.2f}%")

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d")
            st.pyplot(fig)

            st.session_state["model"] = model
            st.session_state["vectorizer"] = vectorizer


# ----------------------------------------------------------
# 🔮 LIVE Prediction Page
# ----------------------------------------------------------
elif page == "Live Prediction":
    st.title("🔮 Live Tweet Sentiment Prediction")

    if "model" not in st.session_state:
        st.warning("Train an ML model first in the 'ML Model' page.")
    else:
        user_tweet = st.text_input("Enter a tweet:")

        if user_tweet:
            vec = st.session_state["vectorizer"].transform([user_tweet])
            pred = st.session_state["model"].predict(vec)[0]
            st.info(f"Predicted Sentiment: **{pred.upper()}**")

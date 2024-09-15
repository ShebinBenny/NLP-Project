# Import necessary libraries
import streamlit as st
from newspaper import Article
from transformers import pipeline

# Initialize the summarization and classification pipelines
summarizer = pipeline("summarization")
classifier = pipeline("zero-shot-classification")

# Set the title of the app
st.title("News Article Summarizer and Categorizer")

# Add a description
st.write("Enter the URL of a news article to get a summary and its category.")

# Create an input field for the article URL
url = st.text_input("Enter article URL:")

# Create a function to fetch and summarize the article
def fetch_and_process_article(url):
    try:
        # Scrape the article
        article = Article(url)
        article.download()
        article.parse()
        content = article.text

        # Generate summary
        summary = summarizer(content, max_length=150, min_length=30, do_sample=False)[0]['summary_text']

        # Define possible categories
        candidate_labels = ["politics", "technology", "sports", "entertainment", "business", "health"]

        # Classify the article
        classification = classifier(content, candidate_labels)
        category = classification['labels'][0]

        return summary, category
    except Exception as e:
        return str(e), None

# Process the URL when the user clicks the button
if st.button("Process Article"):
    if url:
        summary, category = fetch_and_process_article(url)
        if category:
            st.write(f"**Summary:**\n{summary}")
            st.write(f"**Category:** {category}")
        else:
            st.write("Error processing the article. Please check the URL.")
    else:
        st.write("Please enter a valid URL.")

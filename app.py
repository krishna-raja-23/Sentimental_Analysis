import streamlit as st

# ✅ FIRST Streamlit command
st.set_page_config(page_title="Sentiment Chatbot", layout="centered")

# 👇 Then the rest of your code
st.title("📊 Sentiment Analysis Chatbot")
import pandas as pd
import pdfplumber
import plotly.express as px
from transformers import pipeline
from collections import Counter

# Load HuggingFace sentiment analysis model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

model = load_model()

# Function to read different file types
def read_file(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            return pd.read_excel(file)
        elif file.name.endswith('.pdf'):
            text = ''
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return pd.DataFrame({'Text': text.split('\n')})
        else:
            st.error("Unsupported file format.")
            return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# Perform sentiment analysis
def analyze_sentiment(df):
    labels = []
    scores = []
    for text in df['Text']:
        text = str(text)  # ✅ Convert everything to string
        if pd.notna(text) and text.strip():
            try:
                result = model(text[:512])[0]  # Limit to 512 characters
                labels.append(result['label'])
                scores.append(result['score'])
            except Exception:
                labels.append('NEUTRAL')
                scores.append(0)
        else:
            labels.append('NEUTRAL')
            scores.append(0)
    df['label'] = labels
    df['score'] = scores
    return df

# Generate summary
def generate_summary(labels):
    counter = Counter(labels)
    total = sum(counter.values())
    if total == 0:
        return "No valid reviews found."
    top_label, top_count = counter.most_common(1)[0]
    percent = (top_count / total) * 100
    return f"Overall, most reviews are '{top_label}' with {percent:.1f}% of the total."

# Streamlit UI
uploaded_file = st.file_uploader("Upload Excel, CSV or PDF file", type=["csv", "xlsx", "pdf"])

if uploaded_file:
    df = read_file(uploaded_file)
    if df is not None:
        # Dynamically handle the column containing text data
        text_column = None

        # Check if 'Text' column exists
        if 'Text' in df.columns:
            text_column = 'Text'
        else:
            # Prompt the user to select the column containing text data
            st.write("Columns in the uploaded file:", df.columns)
            text_column = st.selectbox("Select the column containing text data:", df.columns)

        # Ensure the selected column is valid
        if text_column:
            df = df[[text_column]].rename(columns={text_column: 'Text'})  # Standardize column name to 'Text'
            df.dropna(subset=['Text'], inplace=True)

            st.subheader("📄 Uploaded Data")
            st.dataframe(df.head())

            # Sentiment Analysis
            with st.spinner("Analyzing Sentiments..."):
                result_df = analyze_sentiment(df)

            st.subheader("🔍 Sentiment Results")
            st.dataframe(result_df.head())

            # Chart Selection
            st.subheader("📈 Charts")
            chart_type = st.selectbox("Select the type of chart to display:", 
                                      ["Bar Chart", "Line Graph", "Bubble Chart"])

            # Generate the selected chart
            if chart_type == "Bar Chart":
                bar_chart_data = result_df['label'].value_counts().reset_index()
                bar_chart_data.columns = ['Sentiment', 'Count']  # Rename columns for clarity

                bar_chart = px.bar(bar_chart_data,
                                   x='Sentiment', y='Count',
                                   labels={'Sentiment': 'Sentiment', 'Count': 'Count'},
                                   title="Bar Chart of Sentiments")
                st.plotly_chart(bar_chart)

            elif chart_type == "Line Graph":
                line_chart = px.line(result_df.reset_index(), x='index', y='score', color='label',
                                     title="Line Graph of Sentiment Scores")
                st.plotly_chart(line_chart)

            elif chart_type == "Bubble Chart":
                bubble_chart = px.scatter(result_df.reset_index(), x='index', y='score',
                                          size='score', color='label',
                                          title="Bubble Chart of Sentiment Scores")
                st.plotly_chart(bubble_chart)

            # Summary
            st.subheader("📝 Overall Summary")
            summary = generate_summary(result_df['label'])
            st.success(summary)
        else:
            st.error("No valid column selected for text data.")

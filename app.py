import streamlit as st
import pandas as pd
import nltk
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
from collections import defaultdict
import re

# Function to truncate text to 448 words
def truncate(text, n_tokens=448):
    tokens = text.split()
    if len(tokens) > n_tokens:
        return " ".join(tokens[:n_tokens])
    return text

# Function to generate a word frequency table
def frequency_table(text):
    words = re.findall(r'\w+', text.lower())
    freq_table = defaultdict(int)
    for word in words:
        freq_table[word] += 1
    return freq_table

# Function to score sentences
def score_sentences(tokenized_sentences, freq_table):
    sentence_scores = defaultdict(float)
    for sentence in tokenized_sentences:
        words_in_sentence = re.findall(r'\w+', sentence.lower())
        for word in words_in_sentence:
            if word in freq_table:
                sentence_scores[sentence] += freq_table[word]
        sentence_scores[sentence] /= len(words_in_sentence) if len(words_in_sentence) > 0 else 1
    return sentence_scores

# Function to calculate average score
def avg(sentence_scores):
    return sum(sentence_scores.values()) / (len(sentence_scores) + 1e-10)  # Avoid division by zero

# Function to summarize a document
def summarize(tokenized_sentences, sentence_scores, threshold):
    summary = " ".join([sent for sent in tokenized_sentences if sentence_scores.get(sent, 0) >= threshold])
    return summary

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    # Read the file as bytes
    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in pdf_document:
        text += page.get_text()
    pdf_document.close()
    return text

# Streamlit app layout
st.title("Tamil Text Summarization")

# Text input
text = st.text_area("Enter Tamil text to summarize", height=200)

# Initialize summary variable
summary = ""

if st.button("Summarize"):
    if text.strip():
        # Process text
        truncated_text = truncate(text)
        freq_table = frequency_table(truncated_text)
        tokenized_sentences = sent_tokenize(truncated_text)
        sentence_scores = score_sentences(tokenized_sentences, freq_table)
        threshold = avg(sentence_scores)
        summary = summarize(tokenized_sentences, sentence_scores, threshold)

        st.subheader("Original Text")
        st.write(truncated_text)

        st.subheader("Summary")
        st.write(summary)

    else:
        st.write("Please enter some text.")

# Optional: Upload file for bulk summarization
uploaded_file = st.file_uploader("Or upload a text file or PDF", type=["txt", "pdf"])

if uploaded_file:
    if uploaded_file.type == "text/plain":
        text_data = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        text_data = extract_text_from_pdf(uploaded_file)

    truncated_text = truncate(text_data)
    freq_table = frequency_table(truncated_text)
    tokenized_sentences = sent_tokenize(truncated_text)
    sentence_scores = score_sentences(tokenized_sentences, freq_table)
    threshold = avg(sentence_scores)
    summary = summarize(tokenized_sentences, sentence_scores, threshold)

    st.subheader("Original Text (From File)")
    st.write(truncated_text)

    st.subheader("Summary (From File)")
    st.write(summary)

# # Visualization: Text length before and after summarization
# st.subheader("Summary Length Comparison")

# # Check if summary is defined before plotting
# if summary:
#     lengths_before = [len(text.split())]
#     lengths_after = [len(summary.split())]

#     fig, ax = plt.subplots()
#     ax.boxplot([lengths_before, lengths_after], labels=["Original", "Summary"])
#     ax.set_ylabel("Word Count")
#     st.pyplot(fig)

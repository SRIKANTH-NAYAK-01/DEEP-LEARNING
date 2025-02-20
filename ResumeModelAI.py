# app.py

import os
import re
import pdfplumber
import nltk
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

# Function to preprocess text
def preprocess_text(raw_text):
    cleaned_text = re.sub(r"(page\s+\d+\s+of\s+\d+|page\s+\d+)", "", raw_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"(\n\s*continued\s*\n)", "", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text).strip()

    tokens = word_tokenize(cleaned_text.lower())
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    filtered_tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalnum() and w not in stop_words]
    return " ".join(filtered_tokens)

# Function to extract "Skills" section from text
def extract_skills_section(text):
    skills_pattern = re.compile(r"(skills|technical skills):(.+?)(\n|\Z)", re.IGNORECASE | re.DOTALL)
    match = skills_pattern.search(text)
    if match:
        return preprocess_text(match.group(2))
    return ""

# Function to calculate cosine similarity
def calculate_cosine_similarity(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    cosine_sim_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    return sorted(zip(range(len(resumes)), cosine_sim_scores), key=lambda x: x[1], reverse=True), vectorizer


# Main Streamlit app
def main():
    st.title("Resume Screening AI  🕵🏻‍♂️")

    # Input job description
    job_description = st.text_area("📄 Enter Job Description", height=200)

    # Upload resumes
    uploaded_files = st.file_uploader("💌 Upload Resumes (PDFs)", type="pdf", accept_multiple_files=True)

    # Button to process resumes
    if st.button("🔍 Process and Rank Resumes"):
        if uploaded_files and job_description:
            resumes_text = []
            resume_names = []
            skills_keywords_list = []

            with st.spinner("Processing resumes..."):
                for uploaded_file in uploaded_files:
                    file_path = f"temp_{uploaded_file.name}"
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Extract and preprocess text
                    extracted_text = extract_text_from_pdf(file_path)
                    if extracted_text:
                        preprocessed_text = preprocess_text(extracted_text)
                        resumes_text.append(preprocessed_text)
                        resume_names.append(uploaded_file.name)

                        # Extract keywords from the skills section
                        skills_keywords = extract_skills_section(extracted_text)
                        skills_keywords_list.append(skills_keywords.split())
                    os.remove(file_path)

            if resumes_text:
                # Calculate cosine similarity
                job_description_processed = preprocess_text(job_description)
                ranked_resumes, vectorizer = calculate_cosine_similarity(job_description_processed, resumes_text)

                # Display rankings in a clean table
                st.subheader("🙂‍↔️ Resume Rankings:")
                rankings = []
                for rank, (index, score) in enumerate(ranked_resumes, start=1):
                    rankings.append({
                        "Rank": rank,
                        "Resume": resume_names[index],
                        "Similarity Score (%)": f"{score * 100:.2f}",
                    })
                rankings_df = pd.DataFrame(rankings)
                st.dataframe(rankings_df)

                # Show matched keywords for the top resume
                top_ranked_index = ranked_resumes[0][0]
                top_resume = resumes_text[top_ranked_index]
                matched_keywords = list(set(word for word in job_description.split() if word in top_resume.split()))
                st.subheader(f"🚀 Key Skills/Features Matched: {resume_names[top_ranked_index]}")
                st.write(f"Matched Keywords: {', '.join(matched_keywords)}")
            else:
                st.warning("⚠️ No resumes were processed. Please try again.")


if __name__ == "__main__":
    main()

  # Footer
st.markdown("""
    <hr>
    <p style="text-align: center; font-size: 14px;"> Developed by Srikanth ❤️</p>
    """, unsafe_allow_html=True)


import streamlit as st
import joblib
import re

st.set_page_config(page_title="Mini ATS", layout="wide")

# Load model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text

st.title("📄 Mini ATS - Resume vs Job Matcher")

col1, col2 = st.columns(2)

with col1:
    resume = st.text_area("Resume", height=300)

with col2:
    jd = st.text_area("Job Description", height=300)

if st.button("Analyze"):
    if resume and jd:
        text = clean_text(f"{resume} SEP {jd}")
        vec = vectorizer.transform([text])

        pred = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]

        st.subheader(f"🎯 Prediction: {pred}")

        st.write("### Confidence:")
        for cls, p in zip(model.classes_, probs):
            st.write(f"{cls}: {p:.2f}")
    else:
        st.warning("Enter both inputs!")


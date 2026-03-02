import streamlit as st
import pickle
import re
import unicodedata
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load model
model = pickle.load(open("food_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text)
    text = unicodedata.normalize('NFKD', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess(text):
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

st.set_page_config(page_title="Food Category Classifier")

st.title("🍽 Food Category Classification")
st.write("Masukkan deskripsi makanan untuk mengetahui kategorinya.")

user_input = st.text_area("Food Description")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Input tidak boleh kosong")
    else:
        cleaned = clean_text(user_input)
        processed = preprocess(cleaned)
        vectorized = vectorizer.transform([processed])
        
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized).max()

        st.success(f"Predicted Category: {prediction}")
        st.info(f"Confidence: {probability:.2f}")
import streamlit as st
import pickle

# Load vectorizer and model
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("Multilingual Language Detection 🌐")
st.write("Enter text and detect its language.")

user_input = st.text_area("Enter your text here:")

def detect_language(text):
    text = str(text).lower()
    text_vector = vectorizer.transform([text])
    return model.predict(text_vector)[0]

if st.button("Detect Language"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        prediction = detect_language(user_input)
        st.success(f"Predicted Language: **{prediction}**")

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

## Load word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key,value in word_index.items()}

# Load the pre-trained model
model = load_model("my_model.keras")


# function to decode review
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# function to preprocess review
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


import streamlit as st
# Streamlit app
st.title("Imdb Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its Positive or Negative.")


# Input text area for the review
user_input = st.text_area("Enter your review here:")
# Button to trigger prediction
if st.button("Predict Sentiment"):
    
    preprocessed_input = preprocess_text(user_input)

    # Make prediction
    prediction = model.predict(preprocessed_input)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'

    # Display the result
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Score: {prediction[0][0]}")

else:
    st.write("Please enter a review and click the button to predict sentiment.")
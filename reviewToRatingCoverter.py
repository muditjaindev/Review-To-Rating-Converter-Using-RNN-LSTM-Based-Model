# It includes the final interface and implementation of our model that converts review to rating

import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import numpy as np
# Load model
model = tf.keras.models.load_model("reviewToRatingModel.h5")

# Load tokenizer
with open("Reviewtokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Max length used while training
max_len = 100  # Make sure this matches what you used during training

st.title("📱 Review To Rating Converter")

st.write("Developed by Mudit | Teerthraj | Tanishq | Jay Shah")

# Input box
user_input = st.text_area("Enter your review:", height=150, placeholder="e.g. This product was amazing!")

# Predict button
if st.button("Predict To Rating"):
    if user_input.strip() == "":
        st.warning("Please enter some review text.")
    else:
        # Preprocess input
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=max_len, padding='post')
        
        # Predict
        prediction = model.predict(padded)
        predicted_index = np.argmax(prediction)
        predicted_rating = predicted_index + 1

        # Output
        st.success(f"Predicted Rating: ⭐ {predicted_rating}")

# Run this file using
# python -m streamlit run reviewToRatingCoverter.py

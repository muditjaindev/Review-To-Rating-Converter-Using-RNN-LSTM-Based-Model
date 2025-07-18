# ⭐ Review to Rating Converter

This project is an end-to-end machine learning interface that predicts a star rating (1–5) based on a user's product review text using a trained deep learning model.

---

## 🔍 Project Overview

**Goal:** Convert free-form product reviews into numerical star ratings using NLP and Deep Learning.

**How It Works:**

- The user types in a review.
- The model processes the text using a pre-trained tokenizer and pads it to the required length.
- A trained neural network predicts a star rating from 1 to 5.
- The predicted rating is displayed with a friendly UI built in Streamlit.

---

## 🚀 Technologies Used

- **TensorFlow / Keras** – for building and training the deep learning model.
- **Natural Language Processing (NLP)** – for text preprocessing and tokenization.
- **Streamlit** – for creating a simple, interactive web interface.
- **Pickle** – for loading the saved tokenizer.
- **NumPy** – for handling numerical predictions.

---

## 🧠 Model Info

- **Model Type**: Deep Neural Network (DNN)
- **Input**: User-written product review
- **Output**: Predicted Rating (1 to 5 stars)
- **Training Data**: Labeled review-rating dataset (not included in this repo)
- **Max Sequence Length**: 100 (used during tokenization and padding)

---

## 👨‍💻 How to Run

### 1. Install Dependencies

```bash
pip install tensorflow streamlit numpy
```

### Run The App
```bash
streamlit run reviewToRatingCoverter.py
```

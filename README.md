Review-To-Rating-Converter-Using-RNN-LSTM-Based-Model
This repository contains a project that leverages Recurrent Neural Networks (RNNs) with Long Short-Term Memory (LSTM) layers to convert textual reviews into numerical ratings. The model is trained to understand the sentiment and context of a review and map it to a corresponding rating, which can be useful for various applications like automated feedback analysis, product review summarization, and sentiment analysis.

Table of Contents
Project Overview

Features

Model Architecture

Dataset

Installation

Usage

Training the Model

Using the Converter

Files in this Repository

Contributing

License

Project Overview
The core idea of this project is to bridge the gap between qualitative textual data (reviews) and quantitative numerical data (ratings). By training an RNN-LSTM model on a dataset of reviews paired with their respective ratings, the system learns to infer a rating from new, unseen reviews. This can automate the process of assigning scores, provide insights into customer feedback at scale, and enhance decision-making processes.

Features
Text-to-Rating Conversion: Converts raw text reviews into a predicted numerical rating.

RNN-LSTM Based Model: Utilizes a robust deep learning architecture suitable for sequential data like text.

Pre-trained Tokenizer: Includes a saved tokenizer for consistent text preprocessing.

Modular Code: Separated modules for model training and review conversion.

Streamlit Web Application: Provides an interactive web interface for real-time review-to-rating predictions.

Example Usage: Demonstrates how to train the model and use it for predictions.

Model Architecture
The model is built using a Recurrent Neural Network (RNN) with LSTM layers. LSTM networks are particularly effective for processing sequential data because they can learn long-term dependencies, which is crucial for understanding the context and nuances in natural language. The architecture typically involves:

Embedding Layer: Converts input text (tokenized words) into dense vector representations.

LSTM Layers: One or more LSTM layers to capture sequential patterns and dependencies in the review text.

Dense Layers: Fully connected layers to process the LSTM output and map it to the final rating.

Output Layer: A final layer with an activation function suitable for the rating scale (e.g., sigmoid for 0-1, or linear for continuous ratings, or softmax for classification into discrete rating categories).

Dataset
The Dataset directory is expected to contain the data used for training the model. This dataset should consist of pairs of text reviews and their corresponding numerical ratings. Common formats include CSV or JSON files.

Expected Dataset Format:

Each entry in the dataset should ideally have at least two fields:

review_text: The actual textual content of the review.

rating: The numerical rating associated with the review (e.g., 1 to 5, or 0 to 1).

Installation
To set up the project locally, follow these steps:

Clone the repository:

git clone https://github.com/mj009/Review-To-Rating-Converter-Using-RNN-LSTM-Based-Model.git
cd Review-To-Rating-Converter-Using-RNN-LSTM-Based-Model

Create a virtual environment (recommended):

python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

Install the required libraries:
The project likely uses libraries such as TensorFlow/Keras, NumPy, Pandas, Scikit-learn, and Streamlit. You might need to infer the exact dependencies from ModelTraining.ipynb or reviewToRatingCoverter.py. A requirements.txt file would typically list these, but if not present, you can install common ones:

pip install tensorflow numpy pandas scikit-learn streamlit

Note: You might need to install keras separately if you are using an older version of TensorFlow or tf-keras if using TensorFlow 2.16+.

Usage
Training the Model
The ModelTraining.ipynb Jupyter notebook contains the code for preparing the dataset, building the RNN-LSTM model, training it, and saving the trained model (reviewToRatingModel.h5) and the tokenizer (Reviewtokenizer.pkl).

Ensure your dataset is in the Dataset directory.

Open the Jupyter notebook:

jupyter notebook ModelTraining.ipynb

Run all cells in the notebook to train the model. This will generate reviewToRatingModel.h5 and Reviewtokenizer.pkl in your project directory.

Using the Converter (Streamlit App)
The reviewToRatingCoverter.py file contains the Streamlit application for making predictions.

Make sure you have trained the model and have reviewToRatingModel.h5 and Reviewtokenizer.pkl in your project root.

Run the Streamlit application:

streamlit run reviewToRatingCoverter.py

This command will open a web browser tab with the Streamlit application, where you can input reviews and get real-time rating predictions.

Files in this Repository
Dataset/: Directory containing the training data.

ModelTraining.ipynb: Jupyter notebook for model training, evaluation, and saving.

Reviewtokenizer.pkl: Saved Keras Tokenizer object used for text preprocessing.

reviewToRatingModel.h5: The trained RNN-LSTM model in HDF5 format.

reviewToRatingCoverter.py: Contains the Streamlit web application for review-to-rating conversion.

Brainstorming Project Topics.txt: A text file potentially containing initial ideas or notes for the project.

.gitattributes: Git configuration file.

README.md: This README file.

Contributing
Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please feel free to:

Fork the repository.

Create a new branch (git checkout -b feature/YourFeature).

Make your changes.

Commit your changes (git commit -m 'Add some feature').

Push to the branch (git push origin feature/YourFeature).

Open a Pull Request.

License
This project is open-source and available under the MIT License.

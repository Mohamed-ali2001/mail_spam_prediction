import numpy as np
import pickle
import streamlit as st
from collections import Counter

# Load the trained model from disk for prediction
loaded_model = pickle.load(open('trained_best_model.sav', 'rb'))

def email_to_feature_vector(email_text):
    """
    Converts the input email text into a feature vector.
    The vector counts the occurrences of each word from the training columns in the email.
    This ensures the input matches the format expected by the trained model.
    """
    training_columns = pickle.load(open('training_columns.pkl', 'rb'))
    words = email_text.split()
    word_counts = Counter(words)
    feature_vector = [word_counts.get(word, 0) for word in training_columns]

    return np.array(feature_vector).reshape(1, -1)

def classify_email(email_text):
    """
    Predicts whether the given email text is spam or not spam.
    Returns a human-readable label based on the model's prediction.
    """
    feature_vector = email_to_feature_vector(email_text)
    prediction = loaded_model.predict(feature_vector)
    return 'No Spam' if prediction == 0 else 'Spam'

def main():

    st.title("Spam Email Classifier")
    email_input = st.text_area("Enter the email text:")

    if st.button("Classify"):
        if email_input.strip():
            result = classify_email(email_input)
            st.write(f"Prediction: {result}")
        else:
            st.write("Please enter some email text to classify.")

if __name__ == '__main__':
    main()
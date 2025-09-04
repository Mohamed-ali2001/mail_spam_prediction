# Mail Spam Classification

## Overview

This project demonstrates a complete machine learning workflow for classifying emails as spam or not spam. It includes data exploration, model training and evaluation, and a simple Streamlit web app for real-time email spam detection.

## Dataset

- **Source:** [Kaggle Email Spam Dataset](https://www.kaggle.com/datasets/balakrishnamp/email-spam-classification-dataset)
- **Description:**  
  The dataset contains a collection of emails with extracted numerical features and a target label (`Prediction`), indicating whether each email is spam (`1`) or not spam (`0`). Features are derived from the email content and metadata.

## Project Structure

```
Mail spam classification/
│
├── mail_spammer.ipynb                # Jupyter notebook for data analysis and model training
├── mail_spammer_detecter_app.py      # Streamlit app for spam detection
├── emails.csv                        # Email dataset (downloaded from Kaggle)
├── trained_best_model.sav            # Saved best-performing model
├── training_columns.pkl              # Saved feature columns used for training
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

## Workflow

### 1. Data Exploration & Model Training (`mail_spammer.ipynb`)
- **Import libraries:** pandas, scikit-learn, seaborn, pickle.
- **Load and inspect data:** View dataset structure, summary statistics, and check for missing values.
- **Visualize target variable:** Check class balance between spam and non-spam emails.
- **Prepare data:** Split features and target, train-test split.
- **Model comparison:** Train and evaluate multiple classifiers (Logistic Regression, Naive Bayes, Decision Tree, Random Forest, Gradient Boosting, k-NN, SVM, Dummy Baseline).
- **Save best model:** Persist the best-performing model and feature columns for later use.

### 2. Spam Detection Web App (`mail_spammer_detecter_app.py`)
- **Streamlit UI:** Simple interface for users to input email text.
- **Feature extraction:** Convert input text to feature vector using training columns.
- **Prediction:** Use the trained model to classify the email as spam or not spam.
- **Result display:** Show prediction result to the user.

## How to Run

### 1. Setup

- Clone this repository or download the project folder.
- Install required Python packages using:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Train the Model

- Open `mail_spammer.ipynb` in Jupyter Notebook or VS Code.
- Run all cells to train models, evaluate, and save the best model and feature columns.

### 3. Launch the Streamlit App

- In your terminal, run:
  ```bash
  streamlit run mail_spammer_detecter_app.py
  ```
- Enter email text in the app to get spam classification.

## Files

- **mail_spammer.ipynb:** Main notebook for data analysis, visualization, model training, and evaluation.
- **mail_spammer_detecter_app.py:** Streamlit app for interactive spam detection.
- **emails.csv:** Dataset file (download from Kaggle if not present).
- **trained_best_model.sav:** Serialized best model (created after running the notebook).
- **training_columns.pkl:** List of feature columns used for training (created after running the notebook).
- **requirements.txt:** List of required Python packages.

## Notes

- The feature extraction in the app assumes the model was trained on word counts from the email text. If your model uses different features, adjust the extraction logic accordingly.
- For best results, ensure the dataset and saved model files are present in the project directory.


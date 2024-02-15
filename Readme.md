# Tuberculosis Fake News Detection using NLTK and Machine Learning

This repository contains a machine learning model built with NLTK (Natural Language Toolkit) to detect fake news related to Tuberculosis. Additionally, it provides a Streamlit web application for testing and demonstrating the effectiveness of the model.

## Dataset
The dataset used for training the model consists of a collection of news articles related to Tuberculosis, labeled as either real or fake. The dataset was gathered from reputable sources and labeled by domain experts.

## Preprocessing
Before training the model, the dataset undergoes preprocessing steps including:
- Tokenization: Breaking down the text into individual words or tokens.
- Stopword Removal: Removing common words that do not contribute to the meaning of the text.
- Lemmatization: Converting words to their base or root form to reduce inflectional forms.

## Feature Extraction
For feature extraction, the model utilizes the Bag-of-Words (BoW) approach. This approach represents text data as numerical feature vectors, where each feature represents the frequency of a word in the document.

## Model Training
The model is trained using various machine learning algorithms such as SVM, Logistic Regression, and Random Forest. After training, the model is evaluated using cross-validation techniques to ensure robustness.

## Streamlit App
The Streamlit web application provides a user-friendly interface for testing the model. Users can input a news article related to Tuberculosis, and the model will classify it as either real or fake. The app also displays the probability score associated with the classification.

## Usage
To run the Streamlit app locally, follow these steps:
1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies using `pip install -r requirements.txt`.
4. Run the Streamlit app using `streamlit run TB_app.py`.
5. Access the app in your web browser at the specified URL.

## Future Improvements
- Incorporating more advanced NLP techniques such as word embeddings (e.g., Word2Vec, GloVe) for better representation of text data.
- Experimenting with deep learning models such as LSTM or Transformers for improved classification performance.
- Enhancing the user interface of the Streamlit app with additional features such as visualization of model predictions and explanations.

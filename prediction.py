import joblib
import numpy
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

port_stem = PorterStemmer()
vectorizer = TfidfVectorizer()
transformer = TfidfTransformer()

vocabulary_feat = TfidfVectorizer(vocabulary=joblib.load('vocabulary.sav'))

def predict(Text):
    
    # Load Logistic Regression Model
    model = joblib.load("FND.sav")
    
    # Preprocess Text
    Text = stemming(Text)
    Text = [Text]
    Text = transformer.fit_transform(vocabulary_feat.fit_transform(Text))
    
    # Making Prediction
    prediction = model.predict(Text)
    if(prediction[0] == 0 ):
        result = 'The news is real'
    else:
        result = 'The news is fake'
    return result


def stemming(text):
    stemmed_content = re.sub('[^a-zA-z]', ' ', text)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content
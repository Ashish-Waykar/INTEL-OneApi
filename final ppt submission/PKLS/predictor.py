import pickle
# from skl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import nltk
# nltk.download('stopwords')
# print('Downloaded Stopwords')
from nltk.corpus import stopwords
import re
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors  import KNeighborsRegressor 
from sklearn.linear_model import LogisticRegression
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
stop_words = STOP_WORDS
import string
punctuations = string.punctuation
from sklearn.feature_extraction.text import HashingVectorizer



# from one daal
from daal4py.sklearn.ensemble import RandomForestRegressor as d4prdf
from daal4py.sklearn.neighbors  import KNeighborsRegressor as d4pknr 
from daal4py.sklearn.linear_model import LogisticRegression as d4plr
from daal4py.sklearn.model_selection import _daal_train_test_split as d4ptts
import time






# section 2
nlp = English()
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        
        return [clean_text(text) for text in X]

    def fit(self, X, y, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


def clean_text(text):
   
    return text.strip().lower()


def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = nlp(sentence)

    # here the token is converted into lowercase if it is a Pronoun and if it is not a Pronoun then it is lemmatized and lowercased    
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words using stopword from spacy library and punctuations from string library
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens

# section 3
bow_vector = CountVectorizer(max_features = 100,tokenizer = spacy_tokenizer,ngram_range=(1,2))

xgboost = MultiOutputRegressor(XGBRegressor())
rand_for = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,max_depth=None,random_state=0))
rand_for_d4prfr = MultiOutputRegressor(d4prdf(n_estimators=100,max_depth=None,random_state=0))

knnr = MultiOutputRegressor(KNeighborsRegressor())
d4pknr = MultiOutputRegressor(d4pknr())



# section 4
# loding pipelines
# sikitlearn  KNeighbors forest
# Title
skl_pipe_title_knr = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('tfidf',TfidfTransformer()),
                 ('regressor', knnr)])
# Headline
skl_pipe_headline_knr = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('tfidf',TfidfTransformer()),
                 ('regressor', knnr)])

# daal4py  KNeighbors forest
# Title
d4p_pipe_title_knr = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('tfidf',TfidfTransformer()),
                 ('regressor', d4pknr)])
# headline
d4p_pipe_headline_knr = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('tfidf',TfidfTransformer()),
                 ('regressor', d4pknr)])

# daal4py Random forest
# title
skl_pipe_title_rdf = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('tfidf',TfidfTransformer()),
                 ('regressor', rand_for)])
# headline
skl_pipe_headline_rdf = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('tfidf',TfidfTransformer()),
                 ('regressor', rand_for)])


# daal4py Random forest
# title
d4p_pipe_title_rdf = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('tfidf',TfidfTransformer()),
                 ('regressor', rand_for_d4prfr)])
# headline
d4p_pipe_headline_rdf = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('tfidf',TfidfTransformer()),
                 ('regressor', rand_for_d4prfr)])


new_data = ["Sanders' Economic Plan Torn Apart By Former Clinton, Obama ..."] 

# Load the pipeline from the pickle file
with open('pipe_title_d4pknr.pkl', 'rb') as f:
    loaded_pipeline_knr = pickle.load(f)

# Use the loaded model for prediction
predictions_knr = loaded_pipeline_knr.predict(new_data)

print(predictions_knr)
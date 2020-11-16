# -*- coding: UTF-8 -*-
# Copyright (C) 2018 Jean Bizot <jean@styckr.io>
""" Main lib for nlp_preprocessing Project
"""

# from os.path import split
from nltk.corpus import stopwords
import string
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def clean_text (text):
    """clean text for NLP models
    """
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ') # Remove Punctuation
    lowercased = text.lower() # Lower Case
    tokenized = word_tokenize(lowercased) # Tokenize
    words_only = [word for word in tokenized if word.isalpha()] # Remove numbers
    stop_words = set(stopwords.words('english')) # Make stopword list
    without_stopwords = [word for word in words_only if not word in stop_words] # Remove Stop Words
    lemma=WordNetLemmatizer() # Initiate Lemmatizer
    lemmatized = [lemma.lemmatize(word) for word in without_stopwords] # Lemmatize
    return lemmatized

if __name__ == '__main__':
    # For introspections purpose to quickly get this functions on ipython
    import nlp_preprocessing
    folder_source, _ = split(nlp_preprocessing.__file__)
    df = pd.read_csv('{}/data/data'.format(folder_source), sep=",", header=None, names=['text'])
    clean_text = df.text.apply(clean_text)
    print(' text cleaned')

# -*- coding: UTF-8 -*-

# Import from standard library
import os
import nlp_preprocessing
import pandas as pd
import pytest
# Import from our lib
from nlp_preprocessing.lib import clean_text


def test_clean_text():
    datapath = os.path.dirname(os.path.abspath(nlp_preprocessing.__file__)) + '/data'
    df = pd.read_csv('{}/data'.format(datapath), sep=",", header=None, names=['text'])
    first_cols = ['text']
    assert list(df.columns) == first_cols
    assert df.shape == (5, 1)
    out = df.text.apply(clean_text)
    assert out.shape == (5,)

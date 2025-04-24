import numpy as np
import pandas as pd
import openml

def test_get_data_as_frame():
    # use a small built-in dataset (iris id=61)
    ds = openml.datasets.get_dataset(61)
    X, y, _, _ = ds.get_data(target='class', as_frame=True)

    # X should be a DataFrame without the target column
    assert isinstance(X, pd.DataFrame), "X should be a DataFrame"
    assert 'class' not in X.columns

    # y should be a Series named 'class'
    assert isinstance(y, pd.Series), "y should be a Series"
    assert y.name == 'class'

    # basic shape check
    assert len(X) == len(y)


import os

import numpy as np
import pandas as pd
import pytest

from nb.nb import NaiveBayes

# Resolve the data path relative to this file, so pytest works from any directory
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "documents.txt")


@pytest.fixture
def df():
    """Load the J&M Appendix B training documents, one per line as: label text"""
    rows = []
    with open(DATA_PATH) as f:
        for line in f:
            if line.strip():
                # Split on the first run of whitespace: label, then the rest is the text
                label, text = line.split(maxsplit=1)
                rows.append({"author": label, "text": text.strip()})
    return pd.DataFrame(rows)

# I chose to split the tests into three different functions instead of one with three checks

def test_priors(df):
    model = NaiveBayes(alpha=1).train(df)  # model with add-one smoothing
    # Check to see if priors match known results
    assert np.allclose(model.priors, [2 / 5, 3 / 5])

def test_likelihoods(df):
    model = NaiveBayes(alpha=1).train(df)  # model with add-one smoothing
    # Check to see if likelihoods match known values, as in the J&M worked example
    v = model.vocabulary
    assert np.allclose(model.likelihoods[:, v["predictable"]], [1 / 29, 2 / 34])
    assert np.allclose(model.likelihoods[:, v["no"]], [1 / 29, 2 / 34])
    assert np.allclose(model.likelihoods[:, v["fun"]], [2 / 29, 1 / 34])

def test_likelihoods_sum_to_one(df):
    model = NaiveBayes(alpha=1).train(df)  # model with add-one smoothing
    # Check if sum along axis for each class is one
    assert np.allclose(model.likelihoods.sum(axis=1), 1)

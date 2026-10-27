from nb.nb import NaiveBayes

def test_nb_class():
    model = NaiveBayes(alpha=1).train(df)   # J&M uses add-1 smoothing
    assert np.allclose(model.priors, [...])
    assert np.allclose(model.likelihoods.sum(axis=1), 1)
import numpy as np

class NaiveBayes:
    """Multinomial Naive Bayes classifier with Lidstone smoothing."""

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.vocabulary = None
        self.classes = None
        self.priors = None
        self.likelihoods = None

    def _tokenize(self, text):
        """Whitespace tokenization"""
        return text.split()

    def _vectorize(self, text):
        """Bag-of-words count vector over training data, ignoring unseen words"""
        vector = np.zeros(len(self.vocabulary))
        for token in self._tokenize(text):
            if token in self.vocabulary:
                vector[self.vocabulary[token]] += 1
        return vector

    def train(self, df, text_col="text", label_col="author"):
        """
        Estimates priors and likelihoods from a DataFrame of labeled documents
        :param df: a pandas DataFrame
        :return: self
        """
        # Build vocabulary using self instead of variable
        self.vocabulary = {}
        for text in df[text_col]:
            for token in self._tokenize(text): # using internal method, could use class instead 
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)

        labels = df[label_col].to_numpy()
        self.classes = np.unique(labels)  # sorted, same order as the priors below

        # Priors
        self.priors = df[label_col].value_counts(normalize=True).sort_index().to_numpy()

        # Bag-of-words matrix
        training_matrix = np.array([self._vectorize(text) for text in df[text_col]])

        # Word counts per class, then Lidstone-smoothing
        n_vocab = len(self.vocabulary)
        self.likelihoods = np.zeros((len(self.classes), n_vocab))
        for i, c in enumerate(self.classes):
            counts = training_matrix[labels == c].sum(axis=0)
            self.likelihoods[i] = (counts + self.alpha) / (counts.sum() + self.alpha * n_vocab)

        return self

    def test(self, df, text_col="text"):
        """
        Predicts a class for every document in the DataFrame
        :param df: a pandas DataFrame
        :return: a numpy array of predictions
        """
        log_priors = np.log(self.priors)
        log_likelihoods = np.log(self.likelihoods)
        predictions = []
        for text in df[text_col]:
            scores = log_priors + log_likelihoods @ self._vectorize(text)
            predictions.append(self.classes[np.argmax(scores)])
        return np.array(predictions)
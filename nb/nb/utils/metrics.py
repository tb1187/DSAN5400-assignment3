from sklearn import metrics

def get_metrics(true, preds):
    """
    Takes gold labels and predictions to compute performance metrics
    :param true: array-like object
    :param preds: array-like object
    :return: a tuple of various performance metrics
    """
    accuracy = metrics.accuracy_score(true, preds)
    f1_score = metrics.f1_score(true, preds)
    conf_matrix = metrics.confusion_matrix(true, preds)

    return accuracy, f1_score, conf_matrix
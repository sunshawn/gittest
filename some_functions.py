def accuracy(tp, tn, fp, fn):
    """
    this function is used for calculating accuracy
    accuracy A = (tp+tn)/(tp+tn+fp+fn)

    :param tp: true positive
    :param tn: true negative
    :param fp: false positive
    :param fn: false negative
    :return: the accuracy
    """
    numerator = tp + tn
    denominator = tp + tn + fp + fn

    return numerator / denominator


def precision(tp, fp):
    """
    this function is used for calculating precision
    precision P = tp/(tp+fp)

    :param tp: true positive
    :param fp: false positive
    :return: the precision
    """
    numerator = tp
    denominator = tp + fp

    return numerator / denominator


def recall(tp, fn):
    """
    this function is used for calculating recall
    recall R = tp/(tp+fn)

    :param tp: true positive
    :param fn: false negative
    :return: the recall
    """
    numerator = tp
    denominator = tp + fn

    return numerator / denominator


def f1(tp, fp, fn):
    """
    this function is used for calculating f1
    F1 = 2PR/(P+R)

    :param tp: true positive
    :param fp: false positive
    :param fn: false negative
    :return: the f1
    """
    numerator = 2 * precision(tp, fp) * recall(tp, fn)
    denominator = precision(tp, fp) + recall(tp, fn)

    return numerator / denominator

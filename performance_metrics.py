import numpy as np
from krippendorff import krippendorff
from pandas import DataFrame as df
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score


def display_performance_metrics(trues, predicted, probs, class_list):
    class_metrics, general_metrics, roc = calculate_performance_metrics(trues, predicted, probs, class_list)
    display(class_metrics.round(2))
    display(general_metrics.round(2))


def print_performance_metrics(trues, predicted, probs, class_list):
    class_metrics, general_metrics, roc = calculate_performance_metrics(trues, predicted, probs, class_list)
    print(class_metrics.round(2))
    print(general_metrics.round(2))


def calculate_performance_metrics(trues, predicted, probs, class_list):
    """
    Calculates some performance metrics given a list of ground truth values and a list of predictions to be compared.
    :param trues: list of ground truths
    :param predicted: list of model predictions
    :param probs: list of model predicted probalities
    :param class_list: the set of all possible labels
    :return: a dataframe with class level metrics and a dataframe with general metrics and a one with ROC values
    """
    class_metrics_data = {'recall': recall_score(trues, predicted, average=None),
                          'precision': precision_score(trues, predicted, average=None),
                          'f1': f1_score(trues, predicted, average=None)}
    class_metrics = df(class_metrics_data, index=class_list)

    i_trues = [list(class_list).index(label) for label in trues]
    i_predicted = [list(class_list).index(label) for label in predicted]
    i_set = np.unique(i_trues + i_predicted)

    from sklearn.metrics import roc_auc_score, roc_curve
    fpr, tpr, thresholds = roc_curve(y_true=trues, y_score=probs, pos_label='pathogen disgust')
    roc = df({'fpr': fpr, 'tpr': tpr})

    general_metrics_data = [(roc_auc_score(trues, probs)),
                            accuracy_score(trues, predicted),
                            krippendorff.alpha(reliability_data=[i_trues, i_predicted],
                                               level_of_measurement='nominal', value_domain=i_set)]
    general_metrics = df(general_metrics_data, index=['auc', 'accuracy', 'krippendorff alpha'], columns=['score'])
    return class_metrics, general_metrics, roc

from pathlib import Path

import numpy as np
import pandas as pd

from IPython.core.display_functions import display
from jinja2 import Environment, FileSystemLoader
from krippendorff import krippendorff
from matplotlib import pyplot as plt
from pandas import DataFrame, DataFrame as df
from pandas.io.formats.style import Styler
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix


def display_performance_metrics(trues, predicted, probs, class_list):
    class_metrics, general_metrics, roc, conf_matrix = calculate_performance_metrics(trues, predicted, probs,
                                                                                     class_list)

    formatted_tables = format_tables(class_metrics, general_metrics, conf_matrix)

    for table in formatted_tables:
        display(table)

    if roc is not None:
        display(roc)


def save_performance_metrics(trues, predicted, probs, class_list, folder: Path):
    class_metrics, general_metrics, roc, conf_matrix = calculate_performance_metrics(trues, predicted, probs,
                                                                                     class_list)

    class_metrics_f, general_metrics_f, conf_matrix_f = format_tables(class_metrics, general_metrics, conf_matrix)

    roc_path = 'roc.svg'
    if roc is not None:
        roc_auc = general_metrics.loc['auc', 'score']
        plt.title('Receiver Operating Characteristic')
        plt.plot(roc['fpr'], roc['tpr'], 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(folder / roc_path)

    html_output = make_html(class_metrics_f, conf_matrix_f, general_metrics_f, roc_path)
    with open(folder / 'performance_metrics.html', 'w') as f:
        f.write(html_output)
    with open(folder / 'performance_metrics.csv', 'w') as f:
        f.write(general_metrics.to_csv())


def make_html(class_metrics_f, conf_matrix_f, general_metrics_f, roc_path):
    env = Environment(loader=FileSystemLoader('disgust/templates'))
    template = env.get_template('template.html')
    data = {'cards': [
        {'title': 'General metrics', 'content': general_metrics_f.to_html()},
        {'title': 'Confusion matrix', 'content': conf_matrix_f.to_html()},
        {'title': 'Class metrics', 'content': class_metrics_f.to_html()},
        {'title': 'ROC', 'content': f'<img src="{roc_path}" />'}
    ]}
    html_output = template.render(data)
    return html_output


def format_tables(class_metrics: DataFrame, general_metrics: DataFrame, conf_matrix: DataFrame) -> tuple[Styler]:
    formatted_class_metrics, formatted_general_metrics = [
        table.style.background_gradient(vmin=0, vmax=1, cmap='Greys_r').format('{:.2f}') for table in
        [class_metrics, general_metrics]]
    formatted_conf_matrix = conf_matrix.style.background_gradient(
        vmin=0, vmax=conf_matrix.sum().sum(), cmap='Greys_r').format(
        lambda c: f'{c} ({100 * c / conf_matrix.sum().sum():.0f}%)')

    return formatted_class_metrics, formatted_general_metrics, formatted_conf_matrix


def print_performance_metrics(trues, predicted, probs, class_list):
    class_metrics, general_metrics, roc, conf_matrix = calculate_performance_metrics(trues, predicted, probs,
                                                                                     class_list)
    print(class_metrics.round(2))
    print(general_metrics.round(2))
    # if roc is not None:
    # print(roc)
    print(conf_matrix)


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

    if probs is not None:
        from sklearn.metrics import roc_auc_score, roc_curve
        fpr, tpr, thresholds = roc_curve(y_true=trues, y_score=probs, pos_label='pathogen disgust')
        roc = df({'fpr': fpr, 'tpr': tpr})
        roc_auc = (roc_auc_score(trues, probs))
    else:
        roc = None
        roc_auc = None

    general_metrics_data = [roc_auc,
                            accuracy_score(trues, predicted),
                            krippendorff.alpha(reliability_data=[i_trues, i_predicted],
                                               level_of_measurement='nominal', value_domain=i_set)]
    general_metrics = df(general_metrics_data, index=['auc', 'accuracy', 'krippendorff alpha'], columns=['score'])

    conf_matrix = pd.DataFrame(confusion_matrix(trues, predicted),
                               index=pd.MultiIndex.from_product([['True:'], class_list]),
                               columns=pd.MultiIndex.from_product([['Predicted:'], class_list]))

    return class_metrics, general_metrics[general_metrics['score'].notna()], roc, conf_matrix
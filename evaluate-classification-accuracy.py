#!/usr/bin/env python
# coding: utf-8

"""
Script to calculate and display performance metrics for binary classifier
"""

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import argparse
import sys

argParser = argparse.ArgumentParser(description='Script for evaluating binary classification accuracy')
required = argParser.add_argument_group('required arguments')
required.add_argument("-in", "--input", required=True, help="Path to input CSV file with classified labelled data")
required.add_argument("-tlc", "--truecol", required=True, help="Name of column in input CSV file with ground truth labels")
required.add_argument("-plc", "--predcol", required=True, help="Name of column in input CSV file with predicted labels")

args = argParser.parse_args()

IN_FNAME = str(args.input) # Input filename
GROUNDTRUTH_COLUMN_NAME = str(args.truecol) # groundtruth column name
PREDICTIONS_COLUMN_NAME = str(args.predcol) # predictions column name

def calculate_metrics(labels, predictions):
    """ Calculates performance metrics for binary classifier

        Metrics: precision, recall, f1 score and accuracy

        Parameters
        ----------
        labels: list
            list of input groundtruth labels
        predictions: list
            list of corresponding predicted labels (by the classifier)

        Returns
        -------
            precision, recall, f1 score and accuracy

    """
    report = classification_report(labels, predictions, output_dict=True)
    acc = accuracy_score(labels, predictions)
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']
    f1_score = report['macro avg']['f1-score']
    return precision, recall, f1_score, acc

df = pd.read_csv(IN_FNAME)
groundtruth_labels = df[GROUNDTRUTH_COLUMN_NAME].tolist()
predicted_labels = df[PREDICTIONS_COLUMN_NAME].tolist()
    
# Calculate metrics
precision, recall, f1_score, accuracy = calculate_metrics(groundtruth_labels, predicted_labels)

print('precision: ', precision)
print('recall: ', recall)
print('f1 score: ', f1_score)
print('accuracy: ', accuracy)
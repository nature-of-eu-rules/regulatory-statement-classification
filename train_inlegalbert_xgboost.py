import argparse
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import xgboost
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from performance_metrics import print_performance_metrics

class_names = ['constitutive', 'regulatory']


def main(input_csv_path: Path, use_pca=False, model_name="inlegal"):
    """Load, extract features, train classifier, predict and compute performance metrics."""
    train_texts, test_texts, train_labels, test_labels = load_text_data(input_csv_path)

    base_path = input_csv_path.parent
    train_features = get_features(base_path / f'train_{model_name}_features.npy', train_texts, model_name=model_name)
    test_features = get_features(base_path / f'test_{model_name}_features.npy', test_texts, model_name=model_name)

    if use_pca:
        print('Transforming features using pca...')
        train_features, test_features = pca_transform(test_features, train_features, base_path)

    output_file_base = f'{model_name}{"_pca" if use_pca else ""}_xgboost_classifier'
    classifier = train_classifier(train_features, train_labels)
    model_path = base_path / (output_file_base+'_xgboost_classifier.json')
    print(f'Saving model to {model_path}.')
    classifier.save_model(model_path)

    predict_and_evaluate(train_features, train_labels, class_names, classifier,
                         save_path=base_path / (output_file_base + '_train_predictions.np'))
    predict_and_evaluate(test_features, test_labels, class_names, classifier,
                         save_path=base_path / (output_file_base + '_test_predictions.np'))


def pca_transform(test_features, train_features, base_path):
    pca = PCA()
    pca.fit(train_features)
    pca_explanations_ = pca.explained_variance_ratio_
    from matplotlib import pyplot as plt
    plt.plot([np.sum(pca_explanations_[:i]) for i, _ in enumerate(pca_explanations_)],
             label='Cumulative explained variance')
    plt.savefig(base_path / 'pca_explained_variance.svg')
    train_top_n_components = pca.transform(train_features)
    test_top_n_components = pca.transform(test_features)
    return train_top_n_components, test_top_n_components


def train_classifier(train_features: np.array, train_labels: list[str]) -> xgboost.XGBClassifier:
    """Train xgboost classifier."""
    n_trees = 200
    model = xgboost.XGBClassifier(n_estimators=n_trees)
    print(f'Training XGBoost model with {n_trees} trees ...')
    model.fit(train_features, train_labels)
    return model


def load_text_data(input_csv_path: Path) -> tuple[list[str], list[str], list[str], list[str]]:
    """Load text data from csv and transform and split into train and test sets.

        Returns
        -------
            train_texts, test_texts, train_labels, test_labels - each a list of strings
    """
    LABEL_COLUMN_NAME = 'Regulatory (1) Constitutive (0)'  # groundtruth column name
    CLASSES = {"C": 0, "R": 1}  # 'C' class refers to 'Constitutive', 'R' class refers to 'Regulatory'
    TRAIN_PERC = 0.8  # Train-test split 80-20
    df = pd.read_csv(input_csv_path)

    valid_df = df[df[LABEL_COLUMN_NAME].isin([0, 1])]

    constitutive_df = valid_df[valid_df[LABEL_COLUMN_NAME] == 0]
    regulatory_df = valid_df[valid_df[LABEL_COLUMN_NAME] == 1]

    data = []
    for row in constitutive_df.itertuples():
        data.append({'premise': row[2], 'label': 'C'})
    for row in regulatory_df.itertuples():
        data.append({'premise': row[2], 'label': 'R'})
    training_data, test_data = split_data(data, TRAIN_PERC)  # split data into train/test sets
    train_texts = [example["premise"] for example in training_data]
    test_texts = [example["premise"] for example in test_data]
    train_labels = [CLASSES[example["label"]] for example in training_data]
    test_labels = [CLASSES[example["label"]] for example in test_data]
    return train_texts, test_texts, train_labels, test_labels


def create_features(texts: list[str], model_tag="law-ai/InLegalBERT") -> torch.Tensor:
    """Create features for a list of texts."""
    max_length = 512
    tokenizer = AutoTokenizer.from_pretrained(model_tag)
    model = AutoModel.from_pretrained(model_tag)

    def process_batch(batch: Iterable[str]):
        cropped_texts = [text[:max_length] for text in batch]
        encoded_inputs = tokenizer(cropped_texts, padding='longest', truncation=True, max_length=max_length,
                                   return_tensors="pt")
        with torch.no_grad():
            outputs = model(**encoded_inputs)
        last_hidden_states = outputs.last_hidden_state
        sentence_features = last_hidden_states.mean(dim=1)
        return sentence_features

    dataloader = DataLoader(texts, batch_size=1)  # batch size of 1 was quickest for my development machine
    features = [process_batch(batch) for batch in tqdm(dataloader, desc=f'Creating features')]
    return np.array(torch.cat(features, dim=0))


def get_features(features_path: str, texts: list[str], overwrite_existing_features: bool = False, model_name="inlegal"):
    """Load existing features or compute new ones and save them for reloading them later."""

    if model_name == "inlegal":
        model_tag = "law-ai/InLegalBERT"
    elif model_name == "legal":
        model_tag = "nlpaueb/legal-bert-small-uncased"
    elif model_name == "bert":
        model_tag = "google-bert/bert-base-uncased"
    else:
        raise ValueError(f"Unsupported model type '{model_name}'")
    print(f"Using {model_tag} features.")
    if not Path(features_path).exists() or overwrite_existing_features:
        print(f'No existing features found at {features_path}. Will create them now.')
        np.save(features_path, create_features(texts, model_tag=model_tag))

    features = np.load(features_path)
    print(f'Loaded features for {len(features)} texts from {features_path}.')
    return features


def predict_and_evaluate(features: np.array, labels: list[str], class_names: list[str], model, save_path=None):
    """Create predictions using the model and print performance metrics when compared to a ground truth."""
    predictions = model.predict(features)
    probs = model.predict_proba(features)
    y_true = [class_names[i] for i in np.array(labels)]
    y_pred = [class_names[i] for i in predictions]
    print_performance_metrics(y_true, y_pred, probs[:, 1], class_names)
    print(confusion_matrix(y_pred, y_true))
    if save_path is not None:
        np.save(save_path, y_pred)


def split_data(data, train_p):  # Copied from train-fewshot-classifyer.py
    """ Splits data into training and testing sets

        Parameters
        ----------

        data: list
            list of training data samples. Each data sample is a Python object of the form
            {'premise' : p, 'label': l} where p is a sentence,
            l is the target class label,
        train_p: float
            ratio of data to use for training (remainder is used for testing) - a number between [0..1)

        Returns
        -------
            train data, test data - each a list of data samples as mentioned above

    """
    random.seed(0)  # added
    c_data = []
    r_data = []
    for item in data:
        if item['label'] == 'C':
            c_data.append(item)
        else:
            r_data.append(item)

    if len(c_data) > 0 and len(r_data) > 0:
        c_len = math.ceil(len(c_data) * train_p)
        r_len = math.ceil(len(r_data) * train_p)

        c_idx = list(set(random.sample(range(0, len(c_data)), c_len)))
        r_idx = list(set(random.sample(range(0, len(r_data)), r_len)))

        train = []
        test = []

        for i in range(0, len(c_data)):
            if i in c_idx:
                train.append(c_data[i])
            else:
                test.append(c_data[i])

        for i in range(0, len(r_data)):
            if i in r_idx:
                train.append(r_data[i])
            else:
                test.append(r_data[i])

        return train, test
    else:
        print("You dont have any examples in your training data for one or more of the classes.")
        return [], []


def parse_arguments():
    argParser = argparse.ArgumentParser(
        description='Train xgboost model on inlegal-Bert features to classify English sentences from EU law as either regulatory or non-regulatory')
    required = argParser.add_argument_group('required arguments')
    required.add_argument("-in", "--input", required=True, type=Path, help="Path to input CSV file with training data.")
    argParser.add_argument("-f", "--use-pca", action="store_true", help="Flag to indicate whether to use pca or not")
    argParser.add_argument("-m", "--model-name", default="inlegal", type=str,
                           help="Name of the model being used. Choose from [bert, legal, inlegal].")

    args = argParser.parse_args()
    return args.input, args.use_pca, args.model_name


if __name__ == "__main__":
    input_path, use_pca, model_name = parse_arguments()
    main(input_path, use_pca, model_name=model_name)

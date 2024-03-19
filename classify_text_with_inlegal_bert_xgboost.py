import argparse
from pathlib import Path

import numpy as np
import xgboost

from train_inlegalbert_xgboost import create_features, class_names

models = {}


def classify_texts(texts: list[str], model_path, return_logits: bool = False):
    """Classifies every text in a list of texts using the xgboost model stored in model_path.

    The xgboost model will be loaded and used to classify the texts. The texts however will first be processed by a
    large language model which will do the feature extraction for every text. The classifications of the
    xgboost model will be returned.
    For training the xgboost model, see train_legalbert_xgboost.py.

    Parameters
    ----------
    texts
        A list of strings of which each needs to be classified.
    model_path
        The path to a stored xgboost model
    return_logits
        return the probabilities of the model

    Returns
    -------
        List of classifications, one for every text in the list

    """
    features = create_features(texts)
    if model_path not in models:
        print(f'Loading model from {model_path}.')
        model = xgboost.XGBClassifier()
        model.load_model(model_path)
        models[model_path] = model

    model = models[model_path]
    if return_logits:
        return model.predict_proba(features)
    return model.predict(features)


def parse_arguments():
    argParser = argparse.ArgumentParser(
        description='Classify English sentences from EU law as either regulatory or non-regulatory.')
    required = argParser.add_argument_group('required arguments')
    required.add_argument("-m", "--model_path", required=True, type=Path, help="Path to xgboost model.")
    required.add_argument("-t", "--text", required=True, type=str, help="Some sentence to classify.")
    args = argParser.parse_args()
    return args.model_path, args.text


if __name__ == "__main__":
    model_path, text = parse_arguments()
    probabilities = classify_texts([text], model_path, return_logits=True)[0]
    classification = np.argmax(probabilities)
    print(f'The model classified the text as a {class_names[classification]} statement.'
          f' ({", ".join([c + ": " + str(p) for p, c in zip(probabilities, class_names)])})')

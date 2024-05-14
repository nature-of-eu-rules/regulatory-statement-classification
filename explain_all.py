import argparse
import json
from pathlib import Path

import dianna
import numpy as np
from dianna.utils.tokenizers import SpacyTokenizer
from tqdm import tqdm

from classify_text_with_inlegal_bert_xgboost import classify_texts
from train_inlegalbert_xgboost import load_text_data
from xai_file_convert import load_json_explanations, convert_json_explanations


def explain_texts(input_csv_path):
    num_samples = 1000
    run_dianna = get_dianna_runner(num_samples)

    _train_texts, test_texts, _, _ = load_text_data(input_csv_path)
    results_json_path = Path(f'results_{num_samples}.json')
    if results_json_path.exists():
        results = load_json_explanations(results_json_path)
    else:
        results = {}

    for statement in tqdm(test_texts):
        if statement not in results:
            current_result = run_dianna(statement)
            current_result = [[(a, int(b), c) for a, b, c in cls] for cls in current_result]

            results[statement] = current_result
            with open(results_json_path, 'w') as fp:
                json.dump(results, fp)
            convert_json_explanations(results_json_path, str(results_json_path)+'_0.csv'), str(results_json_path)+'_1.csv')


def get_dianna_runner(num_samples):
    model_path = Path('..') / 'inlegal_xgboost_classifier_xgboost_classifier.json'

    class StatementClassifier:
        def __init__(self):
            self.tokenizer = SpacyTokenizer(name='en_core_web_sm')

        def __call__(self, sentences):
            # ensure the input has a batch axis
            if isinstance(sentences, str):
                sentences = [sentences]

            probs = classify_texts(sentences, model_path, return_proba=True)

            return np.transpose([(probs[:, 0]), (1 - probs[:, 0])])

    model_runner = StatementClassifier()
    num_features = 1000  # top n number of words to include in the attribution map

    def run_dianna(input_text):
        return dianna.explain_text(model_runner, input_text, model_runner.tokenizer,
                                   'LIME', labels=[0, 1], num_samples=num_samples,
                                   num_features=num_features, )

    return run_dianna


def parse_arguments():
    argParser = argparse.ArgumentParser(
        description='Explain sentence classification by a trained model.')
    required = argParser.add_argument_group('required arguments')
    required.add_argument("-in", "--input", required=True, type=Path, help="Path to input CSV file with training data.")
    argParser.add_argument("-m", "--model-name", default="inlegal", type=str,
                           help="Name of the model being used. Choose from [bert, legal, inlegal].")

    args = argParser.parse_args()
    return args.input, args.model_name


if __name__ == "__main__":
    input_path, model_name = parse_arguments()
    explain_texts(input_path)

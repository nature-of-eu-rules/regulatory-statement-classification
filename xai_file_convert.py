import argparse
import json
from pathlib import Path

import pandas as pd


def load_json_explanations(results_json_path: Path):
    with open(results_json_path, 'r') as fp:
        results = json.load(fp)
        print(f'Loaded {len(results)} existing explanations from {results_json_path}.')
    return results

def convert_json_explanations(json_path:Path, csv_path:Path):
    explanations = load_json_explanations(json_path)
    flat = []
    for sentence in explanations:
        for explanation in explanations[sentence]:
            for word, start_index, attribution_class_0 in explanation:
                flat.append([word, sentence, start_index, attribution_class_0])
    df = pd.DataFrame(flat, columns=['word', 'sentence', 'start_index', 'attribution'])
    df.to_csv(csv_path)


def parse_arguments():
    argParser = argparse.ArgumentParser(
        description='Train xgboost model on inlegal-Bert features to classify English sentences from EU law as either regulatory or non-regulatory')
    required = argParser.add_argument_group('required arguments')
    required.add_argument("-in", "--input_json", required=True, type=Path, help="Path to input JSON with explanations.")
    required.add_argument("-out", "--output_csv", required=True, type=Path, help="Path to output csv file.")

    args = argParser.parse_args()
    return args.input_json, args.output_csv


if __name__ == "__main__":
    input_path, output_path = parse_arguments()
    convert_json_explanations(input_path, output_path)

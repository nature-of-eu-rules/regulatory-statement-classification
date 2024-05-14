import argparse
import json
from pathlib import Path

import pandas as pd


def load_json_explanations(results_json_path: Path):
    with open(results_json_path, 'r') as fp:
        results = json.load(fp)
        print(f'Loaded {len(results)} existing explanations from {results_json_path}.')
    return results

def save_explanations(class_, csv_path, explanations):
    flat = []
    for sentence in explanations:
        for class_i, explanation in enumerate(explanations[sentence]):
            for word, start_index, attribution in explanation:
                if class_i == class_:
                    flat.append([word, sentence, start_index, attribution])
                    
    df = pd.DataFrame(flat, columns=['word', 'sentence', 'start_index', 'attribution'])
    df.to_csv(csv_path)
    
def convert_json_explanations(json_path:Path, csv_path_0:Path, csv_path_1:Path):
    explanations = load_json_explanations(json_path)
    save_explanations(0, csv_path_0, explanations)
    save_explanations(1, csv_path_1, explanations)

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

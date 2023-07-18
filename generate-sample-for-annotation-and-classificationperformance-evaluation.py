#!/usr/bin/env python
# coding: utf-8

"""
Script to generate a representative sample of documents to annotate for regulatory / constitutive sentence classification.
The dataset will be used to evaluate a rulebased approach to regulatory sentence classification, as well as
fine-tune and evaluate a fewshot classification model for sentence classification to do regulatory sentence classification.
"""

import pandas as pd
import math
import argparse
import sys
from os.path import exists

argParser = argparse.ArgumentParser(description='EU law sample document generator: generates a sample of EU legislative documents to annotate for regulatory sentence classification')
required = argParser.add_argument_group('required arguments')
required.add_argument("-in", "--input", required=True, help="Path to input CSV file. See output file of 'eu_rules_metadata_extractor.py' in the https://github.com/nature-of-eu-rules/data-extraction repo for the required columns")
required.add_argument("-out", "--output", required=True, help="Path to output CSV file which stores the generated sample (subset of the rows in the input file)")

args = argParser.parse_args()

IN_FNAME = str(args.input) # Input filename
OUT_FNAME = str(args.output) # Output filename

COLUMNS_TO_REMOVE = ['author', 'responsible_body', 'title', 'addressee', 'procedure_code', 'day', 'month', 'date_adoption', 'date_in_force', 'date_end_validity', 'subject_matters', 'eurovoc', 'directory_code']
YEAR_COLUMN_NAME = 'year'
YEAR_START = 2013.0 # start year for selecting documents from
YEAR_END = 2022.0 # end year for selecting documents from
POLICY_AREA_COLUMN_NAME = 'dc_string'

df = pd.read_csv(IN_FNAME) # import metadata file
df = df.drop(COLUMNS_TO_REMOVE, axis=1) # remove columns irrelevant for sampling
df = df[df[YEAR_COLUMN_NAME].between(YEAR_START, YEAR_END)] # limit to year range 2013 - 2022 (PDF data prior to 2013 is unreliable in quality i.e. OCR errors)
filtered_df = df[~df[POLICY_AREA_COLUMN_NAME].isnull()] # only select legislation that has a clear policy area mentioned
filtered_df = filtered_df[~(filtered_df[POLICY_AREA_COLUMN_NAME] == '')] # only select legislation that has a clear policy area mentioned

# Calculate sample size
Z = 1.645  # Z-score for 95% confidence level
p = 0.5  # Expected proportion (assuming 50% for maximum sample size)
ME = 0.03  # Margin of error (3%)
sample_size = math.ceil((Z ** 2 * p * (1 - p)) / (ME ** 2))

# generate sample that is representative of the distribution of values in the columns
sampled_df = pd.DataFrame() # Create an empty dataframe for the sampled rows
for col in df.columns.tolist():
    sampled_df[col] = df[col].sample(n=sample_size, replace=False).values

# write sample dataframe to file
sampled_df.to_csv(OUT_FNAME, index=False)
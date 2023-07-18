# regulatory-statement-classification
Scripts, algorithms and files for classifying sentences in EU legislative documents (in PDF or HTML format) as either regulatory or non-regulatory in nature using the [Institutional Grammar Tool](https://onlinelibrary.wiley.com/doi/10.1111/padm.12711), as well as for evaluating the accuracy of these approaches. This code is developed as part of the [Nature of EU Rules project](https://research-software-directory.org/projects/the-nature-of-eu-rules-strict-and-detailed-or-lacking-bite).

#### Requirements
+ [Python](https://www.python.org/downloads/) 3.9.12+
+ A tool for checking out a [Git](http://git-scm.com/) repository

#### Setup: before running any scripts in this repository

1. Get a copy of the code:

        git clone git@github.com:nature-of-eu-rules/regulatory-statement-classification.git
    
2. Change into the `regulatory-statement-classification/` directory:

        cd regulatory-statement-classification/
    
3. Create new [virtual environment](https://docs.python.org/3/library/venv.html) e.g:

        python -m venv path/to/virtual/environment/folder/
       
4. Activate new virtual environment e.g. for MacOSX users type: 

        source path/to/virtual/environment/folder/bin/activate
        
5. Install required libraries for the script in this virtual environment:

        pip install -r requirements.txt

#### Description of scripts in this repository

```rule-based-classification.py``` and ```rule-based-classification-batch.py```

Given a list of English sentences which originate in EU legislative documents, these scripts apply a rule-based approach using [grammatical dependency parsing](http://nlpprogress.com/english/dependency_parsing.html) and predefined dictionaries to classify the sentences as either regulatory (1) or non-regulatory (0) in nature. The only difference with ```rule-based-classification-batch.py``` is that the classification results are periodically saved to disk after all sentences in documents from a specific year are processed, rather than saving all results for all documents to file only at the end.

###### Input

A CSV file with at least one column with column-header 'sent'. This column should be a list of English sentences that originate in EU legislative documents.

###### Output

Same CSV file as input with additional two columns: 'regulatory_according_to_rule' and 'attribute_according_to_rule' which are the classification results (0 or 1 whether the sentence is regulatory or not) and the name of the entity (called the 'attribute') in the sentence that is being regulated by the regulatory statement, respectively. The accuracy of the attribute extraction from the sentence is not currently measured and, based on cursory analysis, is likely not as high as the classification accuracy itself.

###### Usage

1. Check the command line arguments required to run the script by typing (use analogous instructions for ```rule-based-classification-batch.py```):

        python rule-based-classification.py -h
        
        OUTPUT >
        
        usage: rule-based-classification.py [-h] -in INPUT -out OUTPUT -agts AGENTS

        Regulatory vs. Non-regulatory sentence classifier for EU legislation based on NLP dependency analysis

        optional arguments:
        -h, --help            show this help message and exit

        required arguments:
        -in INPUT, --input INPUT
                                Path to input CSV file. Must have at least one column with header 'sent' containing sentences from EU legislation in English.
        -out OUTPUT, --output OUTPUT
                                Path to output CSV file in which to store the classification results.
        -agts AGENTS, --agents AGENTS
                                Path to JSON file which contains data of the form {'agent_nouns' : [...list of lowercase English word strings, each of which represents an entity with agency...]}. Some example words
                                include 'applicant', 'court', 'tenderer' etc.

2. Example usage: 

        python rule-based-classification.py --input path/to/input.csv --output path/to/output.csv --agents path/to/agent_nouns.json

```generate-sample-for-annotation-and-classificationperformance-evaluation.py```

Given a list of metadata for EU legislative documents in CSV format (please see [this repo](https://github.com/nature-of-eu-rules/data-extraction) for scripts for downloading such data), this script generates a representative sample, based on variance by year and policy area for the input documents. This sample can be used for human labelling for training a classification model and evaluating the accuracy of this model and the rule-based algorithm implemented in ```rule-based-classification.py```.

###### Input

See metadata output of the ```eu_rules_metadata_extractor.py``` script in [this repo](https://github.com/nature-of-eu-rules/data-extraction). The input CSV file for ```generate-sample-for-annotation-and-classificationperformance-evaluation.py``` should have the same format.

###### Output

A CSV file with rows that are a subset of the rows of the input CSV file with the following columns / metadata retained: ```celex``` (document identifier), ```form``` (legislation type), ```year``` (year when legislation was published), ```dc_string``` (policy area), ```format``` (file extension i.e., PDF/HTML). See [this repo](https://github.com/nature-of-eu-rules/data-extraction) for more info.

###### Usage

1. Check the command line arguments required to run the script by typing:

        python generate-sample-for-annotation-and-classificationperformance-evaluation.py -h
        
        OUTPUT >
        
        usage: generate-sample-for-annotation-and-classificationperformance-evaluation.py [-h] -in INPUT -out OUTPUT

        EU law sample document generator: generates a sample of EU legislative documents to annotate for regulatory sentence classification

        optional arguments:
        -h, --help            show this help message and exit

        required arguments:
        -in INPUT, --input INPUT
                                Path to input CSV file. See output file of 'eu_rules_metadata_extractor.py' in the https://github.com/nature-of-eu-rules/data-extraction repo for the required columns
        -out OUTPUT, --output OUTPUT
                                Path to output CSV file which stores the generated sample (subset of the rows in the input file)

2. Example usage: 

        python generate-sample-for-annotation-and-classificationperformance-evaluation.py --input path/to/input.csv --output path/to/output.csv

```train-fewshot-classifier.py```

Given a labelled dataset of sentences labelled by human legal experts, this script fine-tunes a pre-trained [fewshot](https://arxiv.org/abs/1904.04232) model for the same sentence classification task that ```rule-based-classification.py``` tackles.

###### Input

An input CSV file with training data. At least two columns are required: 1) a column with all items to classify (in our case, English sentences). 2) a column with human-specified labels (either 0 or 1 integer) for each item. In our case, a 0 corresponds to a non-regulatory sentence, and 1 corresponds to a regulatory sentence.

###### Output

+ One or more classification models (depending on what parameters have been specified when running the script), saved to disk as .model files. 
+ A CSV file with validation results and predicted labels for the classification task on the input data. Each row in the file is the classification result for those specific values for the input parameters of the script.

###### Usage

1. Check the command line arguments required to run the script by typing:

        python train-fewshot-classifier.py -h
        
        OUTPUT >
        
        usage: train-fewshot-classifier.py [-h] -in INPUT -ic ITEMSCOL -cc CLASSCOL -b BSIZE -e EPOCHS -t TSPLIT -out OUTPUT

        Fine-tune facebook/bart-large-mnli fewshot model to classify English sentences from EU law as either regulatory or non-regulatory

        optional arguments:
        -h, --help            show this help message and exit

        required arguments:
        -in INPUT, --input INPUT
                                Path to input CSV file with training data.
        -ic ITEMSCOL, --itemscol ITEMSCOL
                                Name of column in input CSV file which contains the items to classify
        -cc CLASSCOL, --classcol CLASSCOL
                                Name of column in input CSV file which contains the classified labels for the items
        -b BSIZE, --bsize BSIZE
                                List of batch sizes e.g. [8,16,32]
        -e EPOCHS, --epochs EPOCHS
                                List of numbers indicating different training iterations or epochs to try e.g. [20,25,30]
        -t TSPLIT, --tsplit TSPLIT
                                Proportion of data to use as training data (the remainder will be used for validation). Number between 0 and 1. E.g. a value of 0.8 means 80 percent of the data will be used for training
                                and 20 for validation.
        -out OUTPUT, --output OUTPUT
                                Path to output CSV file with a summary of training results
2. Example usage: 

        python train-fewshot-classifier.py --input path/to/input.csv --itemscol 'item_col_name' --classcol 'itemlabel_col_name' --bsize '[8,16]' --epochs '[20,25]' --tsplit 0.8 --output path/to/output.csv

```evaluate-classification-accuracy.py```

Given an input CSV file with at least two columns: 1) the list of ground truth binary classification labels (either 0 or 1 integers) and 2) an analogous column for predicted labels, this script computes basic classification performance metrics: precision, recall, F1-score, accuracy (1 - error rate)

###### Output

The classification results printed to console output.

###### Usage

1. Check the command line arguments required to run the script by typing:

        python evaluate-classification-accuracy.py -h
        
        OUTPUT >
        
        usage: evaluate-classification-accuracy.py [-h] -in INPUT -tlc TRUECOL -plc PREDCOL

        Script for evaluating binary classification accuracy

        optional arguments:
        -h, --help            show this help message and exit

        required arguments:
        -in INPUT, --input INPUT
                                Path to input CSV file with classified labelled data
        -tlc TRUECOL, --truecol TRUECOL
                                Name of column in input CSV file with ground truth labels
        -plc PREDCOL, --predcol PREDCOL
                                Name of column in input CSV file with predicted labels

2. Example usage: 

        python evaluate-classification-accuracy.py --input path/to/input.csv --truecol 'ground_truth_labels' --predcol 'predicted_labels'

##### License

Copyright (2023) [Kody Moodley, The Netherlands eScience Center](https://www.esciencecenter.nl/team/dr-kody-moodley/)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
#!/usr/bin/env python
# coding: utf-8

"""
Trains few-shot binary text classifier to identify regulatory vs. comnstitutive sentences.
"""
import pandas as pd
import math
import random
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import pickle

IN_FNAME = 'data/training_data_legal_obligations.csv' # Input filename
LABEL_COLUMN_NAME = 'Regulatory (1) Constitutive (0)' # groundtruth column name
PRETRAINED_MODEL = "facebook/bart-large-mnli" # pretrained few-shot model to finetune
CLASSES = {"C": 0, "R": 1} # 'C' class refers to 'Constitutive', 'R' class refers to 'Regulatory'
BATCH_SIZES = [8]
EPOCHS = [25]

def split_data(data, train_p=TRAIN_PERC):
    """ Splits data into training and testing sets

        Parameters
        ----------

        data: list
            list of training data samples. Each data sample is a Python object of the form
            {'premise' : p, 'hypothesis': h, 'label': l} where p is a sentence, 
            l is the target class label, h is a textual hypothesis that follows from the 
            premise p.
        train_p: float
            ratio of data to use for training (remainder is used for testing) - a number between [0..1)

        Returns
        -------
            train data, test data - each a list of data samples as mentioned above

    """
    global TRAIN_PERC
    TRAIN_PERC = train_p
    
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
    
def train_model(data, classes=CLASSES, batch_size=16, epochs=3):
    """ Trains the few-shot binary classification model and saves it to file

            Parameters
            ----------

            data: list
                list of training data samples. Each data sample is a Python object of the form
                {'premise' : p, 'hypothesis': h, 'label': l} where p is a sentence, 
                l is the target class label, h is a textual hypothesis that follows from the 
                premise p.
            classes: list
                List of training classes together with the values / labels they are associated with
            batch_size: int
                How many training data samples are processed before updating the model
            epochs: int
                Number of training iterations

            Returns
            -------
                classifier model

    """

    # Training data
    training_data = data 

    # Initialize the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL)

    # Prepare training examples
    train_texts = [example["premise"] + " " + example["hypothesis"] for example in training_data]
    train_labels = [example["label"] for example in training_data]

    # Encode training examples
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")

    label2id = CLASSES
    train_labels = [label2id[label] for label in train_labels]

    # Convert inputs to PyTorch tensors
    train_inputs = train_encodings["input_ids"]
    train_masks = train_encodings["attention_mask"]
    train_labels = torch.tensor(train_labels)

    # Fine-tune the model on the training data
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # batch_size = 32
    for epoch in range(epochs):  # Adjust the number of epochs as needed
        optimizer.zero_grad()
        # Calculate the total number of samples
        num_samples = len(train_inputs)
        # Calculate the number of batches
        num_batches = (num_samples + batch_size - 1) // batch_size
        # Loop over the batches
        b_idx = 0
        for batch_index in range(num_batches):
            # print("batch ", str(b_idx))
            b_idx += 1
            # Calculate the batch start and end indices
            start_index = batch_index * batch_size
            end_index = min((batch_index + 1) * batch_size, num_samples)
            
            # Extract the batch from the tensors
            batch_input_ids = train_inputs[start_index:end_index]
            batch_attention_mask = train_masks[start_index:end_index]
            batch_labels = train_labels[start_index:end_index]
    
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
            # outputs = model(input_ids=train_inputs, attention_mask=train_masks, labels=train_labels) # original
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    # Create a zero-shot classification pipeline using the fine-tuned model
    classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

    return classifier

def classify_text(classifier, text):
    """ Performs classification on input text

        I.e., returns whether the input text is regulatory or constitutive.

        Parameters
        ----------

        classifier: model
            few-shot classifier model instance
        text: str
            Input text to classify

        Returns
        -------
            target class label for input text according to the classifier

    """
    global CLASSES
    # List of candidate labels
    candidate_labels = list(CLASSES.keys())

    # Perform zero-shot classification
    result = classifier(text, candidate_labels)

    # print(result)
    return result['labels'][0]

def save(data, filename):
    """ Saves data to file using pickle

        Parameters
        ----------

        data: binary
            Data to save to file
        filename: str
            desired path to saved model file

    """
    # open a file, where you ant to store the data
    file = open(filename, 'wb')
    # dump information to that file
    pickle.dump(data, file)
    # close the file
    file.close()

def load(filename):
    """ Loads model from file

        Parameters
        ----------
        filename: str
            desired path to saved model file

        Returns
        -------
            few-shot classifier model instance

    """
    file = open(filename, 'rb')
    data = pickle.load(file)
    file.close()
    return data

def lookup_if_correct(text, pred_label):
    """ Checks whether the classifier got the class prediction of a particular text right or not

        Parameters
        ----------
        text: str
            Input text to classify
        pred_label: str
            predicted target class label for input text

        Returns
        -------
            True if the predicted label is correct, False otherwise

    """
    global test_texts
    
    for item in test_texts:
        if item['premise'] == text:
            if pred_label == item['label']:
                return True
            else:
                return False
    return False

def validate_model(classifier, test_data):    
    """ Evaluate the model (do the testing on the test data)

        Parameters
        ----------
        classifier: binary
            A few-shot model / classifier instance
        test_data: list of data objects of the form
                {'premise' : p, 'hypothesis': h, 'label': l} where p is a sentence, 
                l is the target class label, h is a textual hypothesis that follows from the 
                premise p.

        Returns
        -------
            elapsed_time (testing time), 
            ratio of correctly classified examples, 
            pipe-delimited string of groundtruth labels, 
            pipe-delimited string of predicted labels

    """
    total = len(test_data)
    correct = 0

    st = time.time()
    predicted_labels = []
    ground_truth_labels = []
    for text in test_data:
        predicted_label = classify_text(classifier, text['premise'])
        if predicted_label == 'C':
            predicted_labels.append('0')
        elif predicted_label == 'R':
            predicted_labels.append('1')

        ground_truth_label = text['label']
        if ground_truth_label == 'C':
            ground_truth_labels.append('0')
        elif ground_truth_label == 'R':
            ground_truth_labels.append('1')

        c = lookup_if_correct(text['premise'], predicted_label)
        if c:
            correct += 1
        
    et = time.time()
    elapsed_time = et - st

    return elapsed_time,(correct/total),'|'.join(ground_truth_labels),'|'.join(predicted_labels)

# read data from file into dataframe
df = pd.read_csv(IN_FNAME)
# make sure we only look at valid rows (that have either 0 or 1 for regulatory or constitutive)
relevant_df = df[df[LABEL_COLUMN_NAME].isin([0, 1])] 
# split data into constitutive and regulatory rows
constitutive_df = relevant_df[relevant_df[LABEL_COLUMN_NAME] == 0]
regulatory_df = relevant_df[relevant_df[LABEL_COLUMN_NAME] == 1]

# translate data into few-shot training samples
data = []
for row in constitutive_df.itertuples():
    curr_entry = {}
    curr_entry['premise'] = row[2]
    curr_entry['hypothesis'] = "This is a constitutive statement."
    curr_entry['label'] = 'C'
    data.append(curr_entry)
    
for row in regulatory_df.itertuples():
    curr_entry = {}
    curr_entry['premise'] = row[2]
    curr_entry['hypothesis'] = "This is a regulatory statement."
    curr_entry['label'] = 'R'
    data.append(curr_entry)

TRAIN_PERC = 0.8 # Train-test split 80-20
training_texts, test_texts = split_data(data) # split data into train/test sets

# Train the models and obtain the classifiers
import calendar
import time
import os.path

data = []
for batch_size in BATCH_SIZES:
    for epoch in EPOCHS:
        curr_row = []
        row_id = str(batch_size) + '-' + str(epoch) + '-' + str(len(training_texts))
        curr_row.append(row_id)
        curr_row.append(len(training_texts))
        curr_row.append(batch_size)
        curr_row.append(epoch)
        st = time.time()
        modelfilename = 'data/' + str(TRAIN_PERC).replace('.','') + '_' + str(batch_size) + '_' + str(epoch) + '.model'
        model_exists = os.path.isfile(modelfilename)
        classifier = None
        if model_exists:
            classifier = load(modelfilename)
        else:
            classifier = train_model(training_texts, batch_size=batch_size, epochs=epoch)
            save(classifier, 'data/' + str(TRAIN_PERC).replace('.','') + '_' + str(batch_size) + '_' + str(epoch) + '.model')

        et = time.time()
        elapsed_time = et - st
        curr_row.append(elapsed_time)
        validation_time, precision, ground_truth_labels, predicted_labels = validate_model(classifier, test_texts)
        curr_row.append(validation_time)
        curr_row.append(precision)
        curr_row.append(ground_truth_labels)
        curr_row.append(predicted_labels)
        data.append(curr_row)

gmt = time.gmtime()
ts = calendar.timegm(gmt)
results_filename = 'data/results_{}.csv'.format(str(ts))
results_df = pd.DataFrame(data, columns=['row_id', 'trainingset_size', 'batch_size', 'epochs', 'training_time', 'validation_time', 'precision', 'groundtruth_labels', 'predicted_labels'])
results_df.to_csv(results_filename, index=False)
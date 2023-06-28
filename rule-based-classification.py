#!/usr/bin/env python
# coding: utf-8

"""
Script to apply a rule-based algorithm to guess whether particular sentences in an
EU legislative document (as found on EURLEX in PDF / HTML format) can be classified as
either regulatory (containing a legal obligation) vs. constitutive (not containing a legal obligation)
Website: http://eur-lex.europa.eu/
"""

import pandas as pd
import spacy
import os

# Load the English language model
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('merge_noun_chunks')

IN_FNAME = "test_data.csv" # Input filename
OUT_DIR = "output-result/" # Output directory
OUT_FNAME = "legal_obl_rulebased_evaluation.csv" # Output filename
DEONTICS = ['shall ', 'must ', 'shall not ', 'must not '] # List of relevant deontic phrases
KNOWN_ATTR = ['director', 'directorate-general', 'director-general', 'directorates-general', 'member states', 'member state'] # known dictionary of attributes for legal obligations

def check_out_dir(data_dir):
    """Check if directory for saving extracted text exists, make directory if not 

        Parameters
        ----------
        data_dir: str
            Output directory path.

    """

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
        print(f"Created saving directory at {data_dir}")

# BEGIN: function definitions
def extract_verb_with_aux(sent, input_word):
    global nlp
    doc = nlp(sent)
    # Find the input word in the sentence
    tokens = []
    for t in doc:
        if t.text == input_word:
            token = t
            tokens.append(token)
    
    if len(tokens) == 0:
        return []
    
    verbs = []
    
    for token in tokens:
        for parent in token.ancestors:
            if parent.pos_ in ['VERB', 'AUX']: # AUX verbs are also possible e.g. "shall be"
                verb = parent.text
                verbs.append(verb)
    
    return verbs

def traverse_ancestors_recursive(token, results_s, results_o):
    # Base case: No more ancestors to traverse
    if not token.ancestors:
        return
    
    # Traverse ancestors recursively until 'nsubj' is found or no more ancestors are left
    for ancestor in token.ancestors:
        # print('ancestor: ', ancestor, ' dep: ', ancestor.dep_, ' pos: ', ancestor.pos_)
        if ancestor.dep_ == 'nsubj' or ancestor.dep_ == 'nsubjpass':
            results_s.append({'text' : ancestor.text, 'pos' : ancestor.pos_})
        elif ancestor.dep_ == 'dobj' or ancestor.dep_ == 'pobj':
            results_o.append({'text' : ancestor.text, 'pos' : ancestor.pos_})
        traverse_ancestors_recursive(ancestor, results_s, results_o)
        
def traverse_children_recursive(token, results_s, results_o):
    # Base case: No more ancestors to traverse
    if not token.children:
        return
        
    # Traverse ancestors recursively until 'nsubj' is found or no more ancestors are left
    for child in token.children:
        # print('child: ', child, ' dep: ', child.dep_, ' pos: ', child.pos_)
        if child.dep_ == 'nsubj' or child.dep_ == 'nsubjpass':
            results_s.append({'text' : child.text, 'pos' : child.pos_})
        elif child.dep_ == 'dobj' or child.dep_ == 'pobj': 
            results_o.append({'text' : child.text, 'pos' : child.pos_})
        traverse_children_recursive(child, results_s, results_o)

def get_subjects_and_objects(sentence, input_word):
    """ Traverses dependency tree to find subjects or objects associated with input deontic (for the legal obligation)

        Parameters
        ----------

        sentence: str
            Input sentence
        input_word: str
            Deontic phrase

        Returns
        -------

            List of subjects / objects associated with the verb(s) associated with the input deontic phrase

    """
    global nlp
    
    verbs = extract_verb_with_aux(sentence, input_word)
    if len(verbs) == 0:
        return -1, -1
    
    doc = nlp(sentence)
    
    subjs = []
    objs = []
    for verb in verbs:
        # Find the input word in the sentence
        token = None
        for t in doc:
            if t.text == verb.lower():
                token = t
                break

        if token is None:
            return [], []

        results_s_a = []
        results_o_a = []
        results_s_c = []
        results_o_c = []
        traverse_ancestors_recursive(token, results_s_a, results_o_a)
        traverse_children_recursive(token, results_s_c, results_o_c)
        subjs.extend((results_s_a + results_s_c))
        objs.extend((results_o_a + results_o_c))
        
    return subjs, objs

def get_deontic_type(sent, deontics=DEONTICS):
    """ Identifies which deontic words appear in a given sentence.

        Parameters
        ----------
        
        sent: str
            Input sentence
        deontics: list
            List of deontic words or phrases
        
        Returns
        -------
            Pipe-delimited string of deontic phrases in the sentence

    """
    global DEONTICS
    result = []
    for deontic in deontics:
        if deontic in (" ".join(sent.split())):
            result.append(deontic)
    if len(result) == 0:
        return 'None'
    else:
        return ' | '.join(result)
    
def contains_kw(stri, listofwords):
    """ Checks if any one of a list of phrases appears in a given string

        Parameters
        ----------
        
        stri: str
            Input string
        listofwords: list
            List of phrases
        
        Returns
        -------
            True if at least one of the phrases in the list appears in the input string, False otherwise

    """
    for item in listofwords:
        if item in stri:
            return True
        
    return False
            
def is_regulatory_or_constitutive(sent):
    """ A rule-based algorithm based on grammatical dependency parsing to guess whether
      an input sentence contains a regulatory statement or (in the converse case - constitutive).

        Parameters
        ----------
        
        sent: str
            Input string
        
        Returns
        -------
            dict {'pred' : val, 'attr' : attr} where 'val' is either a 0 (constitutive) or 1 (regulatory)
            and 'attr' is the algorithm's guess at the name of the entity being regulated 
            in the input sentence (if any)
    """
    global nlp
    global KNOWN_ATTR
        
    deontic_types = get_deontic_type(sent)
    input_word = ''
    if 'shall ' in deontic_types.split(' | ') or 'shall not ' in deontic_types:
        input_word = 'shall'
    else:
        input_word = deontic_types[0].strip().split()[0].strip()
    
    subjs, objs = get_subjects_and_objects(sent, input_word)
 
    if subjs == -1:
        # cannot be a regulatory sentence
        return {'pred' : 0, 'attr' : ''}
    else:
        propns = []
        nonpropns = []
        # 1. Best case: proper noun subject
        for item in subjs:
            if item['pos'] == 'PROPN' or contains_kw(item['text'].lower(), KNOWN_ATTR):
                propns.append(item['text'])
            
        if len(propns) > 0:
            return {'pred' : 1, 'attr' : '|'.join(propns)}
            
        # #  2. 2nd best case: non-proper noun subject
        # for item in subjs:
        #     nonpropns.append(item['text'])
    
        # if len(nonpropns) > 0:
        #     return {'pred' : 1, 'attr' : '|'.join(nonpropns)}            
    
        # # 3. 3rd best case: proper noun object
        # for item in objs:
        #     if item['pos'] == 'PROPN':
        #         propns.append(item['text'])
            
        # if len(propns) > 0:
        #     return {'pred' : 1, 'attr' : '|'.join(propns)}
    
        # # 4. 4th best case: non-proper noun object
        # for item in objs:
        #     nonpropns.append(item['text'])
    
        # if len(nonpropns) > 0:
        #     return {'pred' : 1, 'attr' : '|'.join(nonpropns)}
            
        return {'pred' : 0, 'attr' : ''}

# END: function definitions
# BEGIN: apply classifier to input sentences

check_out_dir(OUT_DIR)
df = pd.read_csv(IN_FNAME)
rule_predictions = []
attributes = []
for index, row in df.iterrows():
    print(index, '/', len(df))
    prediction = is_regulatory_or_constitutive(row['sent'])
    rule_predictions.append(prediction['pred'])
    attributes.append(prediction['attr'])

# extend dataframe with classifier predictions in two new columns   
df['regulatory_according_to_rule'] = rule_predictions
df['attribute_according_to_rule'] = attributes

# write dataframe result to file
df.to_csv(OUT_DIR + OUT_FNAME, index=False)

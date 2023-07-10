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
import json
import argparse
import sys
from os.path import exists

# Load the English language model
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('merge_noun_chunks')

argParser = argparse.ArgumentParser(description='Regulatory vs. Non-regulatory sentence classifier for EU legislation based on NLP dependency analysis')
required = argParser.add_argument_group('required arguments')
required.add_argument("-in", "--input", required=True, help="Path to input CSV file. Must have at least one column with header 'sent' containing sentences from EU legislation in English.")
required.add_argument("-out", "--output", required=True, help="Path to output CSV file in which to store the classification results.")
required.add_argument("-agts", "--agents", required=True, help="Path to JSON file which contains data of the form {'agent_nouns' : [...list of lowercase English word strings, each of which represents an entity with agency...]}. Some example words include 'applicant', 'court', 'tenderer' etc.")

args = argParser.parse_args()

IN_FNAME = str(args.input) # Input filename
OUT_FNAME = str(args.output) # Output filename
AGENT_DICT_FILE = str(args.agents) # Agents dictionary JSON file (data inside must be of the form {'agent_nouns' : [...list of English word strings, each of which represents an entity with agency...]})
DEONTICS = ['shall ', 'must ', 'shall not ', 'must not '] # List of relevant deontic phrases
EXCLUDED_PHRASES = ["shall apply", "shall mean", "this regulation shall apply", "shall be binding in its entirety and directly applicable in the member states", "shall be binding in its entirety and directly applicable in all member states", "shall enter into force", "shall be based", "within the meaning", "shall be considered"] # if these phrases occur in a sentence it means it must be constitutive
EXCLUDED_START_PHRASES = ['amendments to decision', 'amendments to implementing decision', 'in this case,', 'in such a case,', 'in such cases,'] # if these phrases occur at the START of a sentence, it must be constitutive
EXCLUDED_ATTR = ["Directive", "Directives", "Protocol", "Protocols", "Decision", "Decisions", "Paragraph", "Paragraphs", "Article", "Articles", "Agreement", "Agreements", "Annex", "Annexes", "ID", "IDs", "Certification", "Certifications", "Fund", "Funds", "PPE", "Regulation", "Regulations", "CONFIDENTIEL UE/EU CONFIDENTIAL", "instrument", "instruments", "signature", "signatures", "safeguard"] # these phrases can never be part of an attribute (agent being regulated) name
START_TOKENS = ['Article', 'Chapter', 'Section', 'ARTICLE', 'CHAPTER', 'SECTION', 'Paragraph', 'PARAGRAPH'] # tokens at the start of a sentence that can be pruned (indicates that the sentenciser did not break up text into clean sentences)
KNOWN_ATTR = {'director' : 'Director', 'directorate-general' : 'Directorate-General', 'director-general' : 'Director-General', 'directorates-general' : 'Directorates-General', 'member states' : 'Member States', 'member state' : 'Member State'} # known attributes and phrasing

with open(AGENT_DICT_FILE) as json_file: # import list of agent nouns (manually curated subset from ConceptNet: https://github.com/commonsense/conceptnet5/wiki/Downloads)
    AGENT_NOUNS = json.load(json_file)['agent_nouns']

# BEGIN: function definitions
def extract_verb_with_aux(sent, input_word):
    global nlp
    doc = nlp(sent)
    # Find the input word in the sentences
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

def traverse_ancestors_recursive(token, results_s, results_o, dep_seq):
    # Base case: No more ancestors to traverse
    if not token.ancestors:
        return
    
    # Traverse ancestors recursively until 'nsubj', 'pobj', 'agent' is found or no more ancestors are left
    for ancestor in token.ancestors:
        if ancestor.dep_ == 'agent':
            dep_seq.append('agent')
        if ancestor.dep_ == 'nsubj':
            dep_seq.append('nsubj')
            results_s.append({'text' : ancestor.text, 'pos' : ancestor.pos_})
        elif ancestor.dep_ == 'pobj':
            dep_seq.append('pobj')
            if len(dep_seq) >= 2:
                if (dep_seq[len(dep_seq)-1] == 'pobj') and (dep_seq[len(dep_seq)-2] in ['agent', 'prep']): # consecutive agent->pobj or prep->pobj dependencies
                    results_o.append({'text' : ancestor.text, 'pos' : ancestor.pos_})
        traverse_ancestors_recursive(ancestor, results_s, results_o, dep_seq)
        
def traverse_children_recursive(token, results_s, results_o, dep_seq):
    # Base case: No more children to traverse
    if not token.children:
        return
        
    # Traverse children recursively until 'nsubj', 'pobj', 'agent' is found or no more ancestors are left
    for child in token.children:
        if child.dep_ == 'agent':
            dep_seq.append('agent')
        if child.dep_ == 'nsubj':
            dep_seq.append('nsubj')
            results_s.append({'text' : child.text, 'pos' : child.pos_})
        elif child.dep_ == 'pobj': 
            dep_seq.append('pobj')
            if len(dep_seq) >= 2:
                if (dep_seq[len(dep_seq)-1] == 'pobj') and (dep_seq[len(dep_seq)-2] in ['agent', 'prep']): # consecutive agent->pobj or prep->pobj dependencies
                    results_o.append({'text' : child.text, 'pos' : child.pos_})
        traverse_children_recursive(child, results_s, results_o, dep_seq)

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
        return -1, -1, -1
    
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
            return [], [], ''

        dependency_sequence = []
        results_s_a = []
        results_o_a = []
        results_s_c = []
        results_o_c = []
        traverse_ancestors_recursive(token, results_s_a, results_o_a, dependency_sequence)
        traverse_children_recursive(token, results_s_c, results_o_c, dependency_sequence)
        subjs.extend((results_s_a + results_s_c))
        objs.extend((results_o_a + results_o_c))
        
    return subjs, objs, dependency_sequence

def contains_sequence(lst, item1, item2):
    """ Checks whether a two given items appear consecutively in a given list.

            Parameters
            ----------
            
            lst: str
                The input list
            item1:
                A list item of any type that can appear in a Python list
            item2:
                Another list item of any type that can appear in a Python list
            
            Returns
            -------
                True if item1 and item2 appear consecutively (in order) in the input list, False otherwise

    """
    for i in range(len(lst) - 1):
        if lst[i] == item1 and lst[i + 1] == item2:
            return True
    return False

def is_valid_sentence(sent):
    """ Checks whether a given sentence could possibly be regulatory in nature.

        * Certain heuristics can determine that a sentence cannot be regulatory

            Parameters
            ----------
            
            sent: str
                The input sentence
            
            Returns
            -------
                True if sent could possibly be regulatory, False if sent must be constitutive in nature

    """
    global EXCLUDED_PHRASES
    global EXCLUDED_START_PHRASES
    
    for phrase in EXCLUDED_PHRASES:
        if (phrase in sent.lower()) or (phrase in clean_sentence(sent).lower()):
            return False
        
    for start_phrase in EXCLUDED_START_PHRASES:
        if sent.lower().startswith(start_phrase):
            return False
    return True

def get_index_of_next_upper_case_token(sent_tokens, start_index = 3):
    """Gets index of first word (after the given start_index) in list of words
      which starts with an uppercase character.

        Parameters
        ----------
        sent_tokens: list
            List of words.
        start_index: int
            the starting index from which the function starts searching

        Returns
        -------
        i: int
            the first index after start_index which has a word starting with an uppercase character

    """
    for i in range(start_index, len(sent_tokens)):
        if sent_tokens[i][0].isupper():
            return i
    return -1
            
def clean_sentence(sent):
    """Formats a sentence to be more easily processed downstream for classifying them as regulatory or not.

        Parameters
        ----------
        sent: str
            The sentence.
        
        Returns
        -------
            The processed sentence.

    """

    global START_TOKENS
    sent_tokens = sent.split()
    if sent_tokens[0].strip() in START_TOKENS:

        if sent_tokens[1].strip().isnumeric():
            if sent_tokens[2].strip()[0].isupper():
                # find position / index of next upper case token in sent
                i = get_index_of_next_upper_case_token(sent_tokens)
                if i > 2:
                    return ' '.join(sent_tokens[i:])
                else:
                    return ' '.join(sent_tokens[3:])
            else:
                return ' '.join(sent_tokens[2:])
        else:
            return ' '.join(sent_tokens)
    else:
        return ' '.join(sent_tokens)

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

def contains_kw_token(stri, listofwords):
    """ Checks if any one of a list of phrases appears in a given string

        Similar to contains_kw() function. Checks if any token within input string appears in list

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

    tokens = stri.split()
    for token in tokens:
        if contains_kw(token, listofwords):
            return True
    return False

def is_agent_token(token):
    """ Checks if given word / token is an agent noun

        Words such as 'manufacturer', 'applicant' indicate agenthood but cannot be recognized by traditional
        NLP NER and POS-tagging algorithms. We use a dictionary approach here from ConceptNet:
        https://github.com/commonsense/conceptnet5/wiki/Downloads

        Parameters
        ----------
        
        token: str
            Input word
        
        Returns
        -------
            True if the input word appears in the predefined agent noun dictionary, False otherwise.

    """   
    global AGENT_NOUNS
    for item in AGENT_NOUNS:
        if item in token.strip():
            return True
    
    return False

def contains_agent_noun(stri):
    """ Checks if given phrase contains agent nouns as either the first or last token

        If the first or last token is an agent noun, chances are that that full phrase indicates an agent noun phrase
        E.g. "applicant of the shipment" (first token "applicant" is an agent noun, makes the entire phrase an agent noun phrase)
        E.g. "legal advisor" (last token "advisor" is an agent noun, makes entire phrase an agent noun phrase)

        Parameters
        ----------
        
        stri: str
            Input noun phrase
        
        Returns
        -------
            True if the input noun phrase is an agent noun phrase, False otherwise.

    """  
    tokens = stri.split()
    
    if len(tokens) > 0:
        first_token = tokens[0]
        if is_agent_token(first_token.lower()):
            return True
        
        if (len(tokens)-1) > 0: # more than 1 token in noun phrase
            last_token = tokens[len(tokens)-1]
            if is_agent_token(last_token.lower()):
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

    if is_valid_sentence(sent):
        sent = clean_sentence(sent)
        
        for item in KNOWN_ATTR:
            sent.replace(item, KNOWN_ATTR[item])
        
        deontic_types = get_deontic_type(sent)
        input_word = ''
        if 'shall ' in deontic_types.split(' | ') or 'shall not ' in deontic_types:
            input_word = 'shall'
        else:
            input_word = deontic_types[0].strip().split()[0].strip()
    
        subjs, objs, depseq = get_subjects_and_objects(sent, input_word)

        if subjs == -1:
            # cannot be a regulatory sentence
            return {'pred' : 0, 'attr' : ''}
        else:
            propns = []
            nonpropns = []

            # Subjs in sentence       
            if len(subjs) > 0:
                
                for item in subjs:
                    if len(item['text'].strip()) > 1:
                        if (item['pos'] == 'PROPN') and not contains_kw_token(item['text'], EXCLUDED_ATTR): # 1. Best case: proper noun subject
                            propns.append(item['text'])
                        if (item['pos'] == 'NOUN') and contains_agent_noun(item['text']) and not contains_kw_token(item['text'], EXCLUDED_ATTR): # 2. Next best case: check if subject is an agent noun (ConceptNet subset)
                            propns.append(item['text'])

                        if len(propns) > 0:
                            return {'pred' : 1, 'attr' : '|'.join(propns)}
                
                
                # 3. Check for 'they' or 'it' subjects
#                 for item in subjs:
#                     if len(item['text'].strip()) > 1:
#                         if (item['pos'] == 'PRON') and item['text'].lower().strip() in ['they', 'it']:
#                             propns.append(item['text'])

#                         if len(propns) > 0:
#                             return {'pred' : 1, 'attr' : '|'.join(propns)}         

            # No subjs in sentence
            else:
                # 3. Third best case: check for passive voice objects (hidden subjects)
                if (contains_sequence(depseq, 'agent', 'pobj') or contains_sequence(depseq, 'prep', 'pobj')):
                    for item in objs:
                        if len(item['text'].strip()) > 1:
                            if (item['pos'] == 'PROPN') and not contains_kw_token(item['text'], EXCLUDED_ATTR):
                                propns.append(item['text'])
                            if (item['pos'] == 'NOUN') and contains_agent_noun(item['text']) and not contains_kw_token(item['text'], EXCLUDED_ATTR): # check if objects are agent nouns
                                propns.append(item['text'])

                            if len(propns) > 0:
                                return {'pred' : 1, 'attr' : '|'.join(propns)} 
                    
            return {'pred' : 0, 'attr' : ''}        
    else:
        return {'pred' : 0, 'attr' : ''}
    
# END: function definitions
# BEGIN: apply classifier to input sentences

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
df.to_csv(OUT_FNAME, index=False)

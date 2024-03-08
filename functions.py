import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer as wnLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords
import re

def preproc(data):
    """
    Preprocesses raw text data for NLP model

    @param: DataFrame data - data to be processed
    @return: processed data
    @rtype: DataFrame
    """
    #print(data.loc[:20])

    data['AllText'] = data['AllText'].str.replace(r'[^a-zA-Z\d\s\-]', '', regex = True)                             # remove non-alphanumeric text (excluding spaces and hyphens)
    data['AllText'] = data['AllText'].str.lower()                                                                   # lowercase all text
    data['AllText'] = data.apply(lambda row: word_tokenize(row['AllText']), axis=1)                                 # tokenize all text into words

    wnl = wnLemmatizer()
    for i,entry in enumerate(data['AllText']):                                                                      # remove all stopwords and lemmatize them into their root word based on part-of-speech
        words = []
        for word,tag in pos_tag(entry):
            if word not in stopwords.words('english'):
                root_word = wnl.lemmatize(word,pos_tagger(tag))
                words.append(root_word)
        data.loc[i,'AllText'] = str(words)

    #print(data.loc[:20])
    return data

def pos_tagger(tag):
    """
    Returns simplified WordNet tag given a NLTK part-of-speech tag

    @param: string tag - NLTK pos tag
    @return: WordNet tag
    @rtype: string
    """

    if tag.startswith('V'):                                                                                         # default pos tag is noun
        return wn.VERB
    elif tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('R'):
        return wn.ADJ
    else:          
        return wn.NOUN
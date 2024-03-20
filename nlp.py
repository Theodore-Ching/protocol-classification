import numpy as np
import pandas as pd
import functions
from sklearn.datasets import fetch_20newsgroups
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

do_tfidf = True                                                                                                                                                                                                 # config
do_count = True
do_NB = True
do_SVM = True

# code for 20 Newsgroups dataset

categories = ['comp.windows.x', 'comp.sys.mac.hardware', 'sci.space', 'sci.med', 'rec.sport.hockey', 'rec.autos', 'rec.motorcycles', 'misc.forsale', 'talk.politics.mideast', 'talk.religion.misc']             # categories for training/testing

train = fetch_20newsgroups(subset='train', categories = categories)                                                                                                                                             # fetch training data
test = fetch_20newsgroups(subset='test', categories = categories)                                                                                                                                               # fetch test data

trainX = train.data
testX = test.data
trainY = train.target
testY = test.target


# code for Yahoo Answers dataset
'''
train = pd.read_csv(r"C:\Stuff\Coding\protocol classification\archive\train.csv", nrows= 1000000, header=None, encoding='latin-1')                                                                               # read raw training data
test = pd.read_csv(r"C:\Stuff\Coding\protocol classification\archive\test.csv", header=None, encoding='latin-1')                                                                                                 # read raw test data
train = functions.join_cols(train, 3)                                                                                                                                                                            # condense all text into joint column
test = functions.join_cols(test, 3)

trainX = train['All']
testX = test['All']
trainY = train["Index"]
testY = test["Index"]
'''

print("# of training documents = ", len(trainX))

if do_tfidf:                                                                                                                                                                                                    # logic for train-testing NB/SVM models using TDIDF/count vectorizers
    vec = TfidfVectorizer()
    train_vec = vec.fit_transform(trainX)
    test_vec = vec.transform(testX)

    #print(vec.vocabulary_)

    if do_NB:
        functions.train_test('NB', 'tdidf', train_vec, test_vec, trainY, testY)

    if do_SVM:
        functions.train_test('SVM', 'tdidf', train_vec, test_vec, trainY, testY)

if do_count: 
    vec = CountVectorizer()
    train_vec = vec.fit_transform(trainX)
    test_vec = vec.transform(testX)

    #print(vec.vocabulary_)

    if do_NB:
        functions.train_test('NB', 'count', train_vec, test_vec, trainY, testY)

    if do_SVM:
        functions.train_test('SVM', 'count', train_vec, test_vec, trainY, testY)

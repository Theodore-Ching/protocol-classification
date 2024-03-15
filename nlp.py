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
data = pd.read_csv(r"C:\Stuff\Coding\protocol classification\archive\train.csv", header=None, encoding='latin-1')                                                                                               # read raw data from csv into DataFrame
data.columns = ['Index', 'Text1', 'Text2', 'Text3']
headers = data.columns[1:]
data[headers] = ' ' + data[headers].astype(str)
data['AllText'] = data[headers].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)                                                                                                                     # condense all text columns into a joint column

trainX, testX, trainY, testY = model_selection.train_test_split(data['AllText'],data['Index'],test_size=0.1)                                                                                                    # set 10% of data as test data and the rest as training data
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

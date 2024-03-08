import numpy as np
import pandas as pd
import functions
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

vec_type = "count"                                                                                                                                  # set vectorizer type (count or tfidf)

data = pd.read_csv(r"C:\Stuff\Coding\protocol classification\archive\train.csv", nrows = 10000, header=None, encoding='latin-1')                    # read 10000 rows of data into DataFrame
data.columns = ['Index', 'Text1', 'Text2', 'Text3']
headers = data.columns[1:]
data[headers] = ' ' + data[headers].astype(str)
data['AllText'] = data[headers].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)                                                         # condense all text columns into a joint column
data = functions.preproc(data)                                                                                                                      # pre-process data

trainX, testX, trainY, testY = model_selection.train_test_split(data['AllText'],data['Index'],test_size=0.1)                                        # split 10% of data into test data and rest into training data

if vec_type == "count":                                                                                                                             # vectorize data (by count or tfidf)
    vec = CountVectorizer(max_features=5000)
else: 
    vec = TfidfVectorizer(max_features=5000)
vec.fit(data['AllText'])
trainX_trans = vec.transform(trainX)
testX_trans = vec.transform(testX)

#print(vec.vocabulary_)

model_NBC = naive_bayes.MultinomialNB()                                                                                                             # Naive Bayes Classifier algorithm
model_NBC.fit(trainX_trans, trainY)

predictions_NBC = model_NBC.predict(testX_trans)

print("Naive Bayes Classifier Accuracy (" + vec_type + ") -> ",accuracy_score(predictions_NBC, testY)*100)

import numpy as np
import pandas as pd
from sklearn import model_selection, naive_bayes, svm, metrics

def train_test(model_type, vec_type, train_vectors, test_vectors, train_target, test_target):
    """
    Trains and tests a specified model using given vectors and targets
    """
    if model_type == 'SVM':                                                                                                                                 # set SVM model 
        model = svm.SVC(C = 1.0, kernel = 'linear', degree = 3, gamma = 'auto')
        model_name = 'Support Vector Machine'
    else:                                                                                                                                                   # set Naive Bayes model (default)
        model = naive_bayes.MultinomialNB()
        model_name = 'Naive Bayes Classifier'
    
    model.fit(train_vectors, train_target)                                                                                                                  # train model
    predictions = model.predict(test_vectors)                                                                                                               # test model
        
    print(model_name + " (" + vec_type + ")")
    #print('{:<18}  {:<18}'.format("accuracy score:", metrics.accuracy_score(predictions, test_target)))                                                     # print accuracy score
    print('{:<18}  {:<18}'.format("f1 score:", metrics.f1_score(test_target, predictions , average='macro')))                                               # print f1 score

    return

def join_cols(df, n):
    """
    Joins n number of columns in a Dataframe (ignores first column)
    """
    df.columns = np.concatenate([["Index"], range(n)])
    headers = df.columns[1:]
    df[headers] = ' ' + df[headers].astype(str)
    df['All'] = df[headers].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)                                                                                                                       # condense all text columns into a joint column
    df = df.drop(headers, axis = 1)

    return df
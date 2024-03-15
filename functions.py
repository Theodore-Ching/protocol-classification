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
        
    print(model_name + " Accuracy (" + vec_type + ") -> " , metrics.f1_score(test_target, predictions , average='macro'), "%")                              # print accuracy score

    return
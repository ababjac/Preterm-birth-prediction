import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
import xgboost as xgb

import helpers
#--------------------------------------------------------------------------------------------------#

def run_SVM(X_train, X_test, y_train, y_test, image_name=None, image_path=None, param_grid=None, label=None, title=None, color=None):

        if param_grid == None:
            param_grid = {
                'C': [0.1, 1, 10, 100, 1000],
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'kernel': ['rbf', 'linear'],
                'probability': [True]
            }

        clf = GridSearchCV(
            estimator=svm.SVC(),
            param_grid=param_grid,
            cv=5,
            n_jobs=5,
            verbose=0
        )

        print('Building model for label:', label)
        clf.fit(X_train, y_train)

        print('Predicting on test data for label:', label)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test) #get probabilities for AUC
        probs = y_prob[:,1]
        
        return y_pred, probs

#--------------------------------------------------------------------------------------------------#

def run_RF(X_train, X_test, y_train, y_test, image_name=None, image_path=None, param_grid=None, label=None, title=None, color=None):

    if param_grid == None:
        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['sqrt', 'log2'],
            'max_depth' : [4,5,6,7,8],
            'criterion' :['gini', 'entropy'],
        }
        
    #kfolds = helpers.undersampledKFold(X_train, y_train)
    kfolds = StratifiedKFold(n_splits=10).split(X_train, y_train)

    clf = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=param_grid,
        cv=kfolds,
        n_jobs=5,
        verbose=0
    )

    print('Building model for label:', label)
    clf.fit(X_train, y_train)

    print('Predicting on test data for label:', label)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test) #get probabilities for AUC
    probs = y_prob[:,1]

    return y_pred, probs
    
#--------------------------------------------------------------------------------------------------#

def run_XGBoost(X_train, X_test, y_train, y_test, image_name=None, image_path=None, label=None, title=None, color=None, param_grid=None):
    
    if param_grid == None:
        param_grid = {
            'objective' : ['binary:logistic'],
            'max_depth' : [6,7,8],
            'n_estimators' : [200, 500]
        }
        
    #kfolds = helpers.undersampledKFold(X_train, y_train)
    kfolds = StratifiedKFold(n_splits=10).split(X_train, y_train)
    
    clf = GridSearchCV(
        estimator=xgb.XGBClassifier(),
        param_grid=param_grid,
        cv=kfolds,
        n_jobs=5,
        verbose=0
    )

    print('Building model for label:', label)
    clf.fit(X_train, y_train)

    print('Predicting on test data for label:', label)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test) #get probabilities for AUC
    probs = y_prob[:,1]
    
    return y_pred, probs

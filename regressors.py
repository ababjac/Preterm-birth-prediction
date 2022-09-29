import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor 
from sklearn import svm
import xgboost as xgb

#import graphing

#--------------------------------------------------------------------------------------------------#

def run_SVM(X_train, X_test, y_train, y_test, image_name=None, image_path=None, param_grid=None, label=None, title=None, color=None):

        if param_grid == None:
            param_grid = {
                'C': [0.1, 1, 10, 100, 1000],
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'kernel': ['rbf', 'linear']
            }

        reg = GridSearchCV(
            estimator=svm.SVR(),
            param_grid=param_grid,
            cv=5,
            n_jobs=5,
            verbose=0
        )

        print('Building model for label:', label)
        reg.fit(X_train, y_train)

        print('Predicting on test data for label:', label)
        y_pred = reg.predict(X_test)
        
        return y_pred

#--------------------------------------------------------------------------------------------------#

def run_RF(X_train, X_test, y_train, y_test, image_name=None, image_path=None, param_grid=None, label=None, title=None, color=None):

    if param_grid == None:
        param_grid = {
            'n_estimators': [200, 500, 1000],
            'max_features': ['sqrt', 'log2'],
            'max_depth' : [4,5,6,7,8],
            'criterion' : ['squared_error', 'poisson']
        }

    reg = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid=param_grid,
        cv=5,
        n_jobs=5,
        verbose=0
    )

    print('Building model for label:', label)
    reg.fit(X_train, y_train)

    print('Predicting on test data for label:', label)
    y_pred = reg.predict(X_test)

    return y_pred
    
#--------------------------------------------------------------------------------------------------#

def run_XGBoost(X_train, X_test, y_train, y_test, image_name=None, image_path=None, label=None, title=None, color=None):
    
    reg = xgb.XGBRegressor(seed=5000)

    print('Building model for label:', label)
    reg.fit(X_train, y_train)

    print('Predicting on test data for label:', label)
    y_pred = reg.predict(X_test)
    
    return y_pred

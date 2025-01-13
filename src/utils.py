import os
import sys

# dill helps to create pickle file
import dill

import numpy as np
import pandas as pd

from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

import pickle

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        #iterate trhough list of all models
        for i in range(len(list(models))):
            model = list(models.values())[i]

            # list down all the params
            para = param[list(models.keys())[i]]


            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            
            # perform fit on xtrain and ytrain
            # model.fit(X_train, y_train)  # Train model

            # prediction on xtrain and xtest
            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            # compute r2 score to evaluate model
            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            # append scores with key and test model score
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    

# loads pickle file using dil

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
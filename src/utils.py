from sklearn.metrics import accuracy_score, f1_score
import os
import dill
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
def evaluate_models(X_train, y_train, X_test, y_test, models, params, classification=False):  
    try:
        report = {}
        for model_name in models:
            model = models[model_name]
            param_grid = params.get(model_name, {})

            gs = GridSearchCV(model, param_grid, cv=3)
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            if classification:
                score = accuracy_score(y_test, y_test_pred)
            else:
                y_train_pred = model.predict(X_train)
                score = r2_score(y_test, y_test_pred)

            report[model_name] = score

        return report   
    except Exception as e:
        logging.error(f"Error in model evaluation: {e}")
        raise CustomException(e, sys)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e   


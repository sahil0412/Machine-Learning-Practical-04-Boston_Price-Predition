import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train,y_train,X_test,y_test,model):
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        print('Training Score : ', model.score(X_train, y_train))
        print('Testing Score  : ', model.score(X_test, y_test))


        print('R^2:',r2_score(y_train, y_pred))
        print('Adjusted R^2:',1 - (1-r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
        print('MAE:',mean_absolute_error(y_train, y_pred))
        print('MSE:',mean_squared_error(y_train, y_pred))
        print('RMSE:',np.sqrt(mean_squared_error(y_train, y_pred)))
        # Predict Testing data
        y_test_pred =model.predict(X_test)
        r2 = r2_score(y_test, y_test_pred)
        mse = mean_squared_error(y_test, y_test_pred)
        return r2, mse, model
    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)
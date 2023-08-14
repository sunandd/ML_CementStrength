import os 
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.utils import save_object
from src.utils import upload_file
from src.utils import evaluate_models

@dataclass
class ModelTrainerCinfig:
    trained_midel_file_path =os.path.join('artifacts', 'midel.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerCinfig()   

    def initate_model_training(self, train_array , test_array):
        try:
            logging.info('splitting dependant and independent variable')
            x_train, y_train ,x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )


            models={
                'LinearRegression' : LinearRegression(),
                'ElasticNet' : ElasticNet(),
                'DecisionTreeRegressor' : DecisionTreeRegressor(),
                'RandomForestRegressor' : RandomForestRegressor(),
                'SVR' : SVR(),
                'XGBRegressor' : XGBRegressor()

            }

            model_report : str = evaluate_models(x_train,y_train,x_test,y_test,models)
            logging.info(f'model_report : {model_report}')

            best_model_score = max(sorted(model_report.values()))

            best_model_name =list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model =models[best_model_name]

            logging.info(f'best model found, model name : {best_model_name}, score :{best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_midel_file_path,
                obj=best_model
            )




        except Exception as e:
            logging.info('Exception occured at model training')
            raise CustomException(e,sys)







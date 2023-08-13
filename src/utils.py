import os 
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging

import boto3
import dill
from sklearn.model_selection import train_test_split


def save_object(file_path, obj):
    try:
        dir_path= os.path.dirname(file_path)
        os.makedir(dir_path, exist_ok=True)
        with open (file_path, "wb") as file_obj:
            dill.jump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)


def upload_file(from_filename, to_filename, bucket_name):
    try:
        s3_resource = boto3.resource("s3")
        s3_resource.meta.client,upload_file(from_filename, bucket_name, to_filename)


    except Exception as e:
        raise CustomException(e,sys) 

def download_model(bucket_name, bucket_file_name, dest_file_name):
    try:
        s3_client = boto3.client("s3")
        s3_client.download_file(bucket_name, bucket_file_name, dest_file_name)

        return dest_file_name
    
    except Exception as e:
        raise CustomException(e,sys)
def evaluate_models(x,y,models):
    try:
        x_train, x_test, y_train, y_test =train_test_split(x, y, train_size=0.35 ,random_state=50) 

        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(x_train, y_train)

            y_test_pred = model.predict(x_test)

            test_model_score = r2_score(y_test,y_test_pred )
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e,sys)    



              
            
import os 
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging

import dill


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
        s3_sorce =boto3.resource("s3")
        s3_resource.meta.client,upload_file(from_filename, bucket_name, to_filename)


    except Exception as e:
        raise CustomException(e,sys)            
            
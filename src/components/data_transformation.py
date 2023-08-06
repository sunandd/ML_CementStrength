import os 
import sys
import pandas as pd
import numpy as numpy
from src.exception  import CustomException
from  src.logger import logging
from src.utils import save_object

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from dataclasses import dataclass



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_c0nfig= DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            replace_duplicates_with_mean= lambda x: np.where(x== '')



        except Exception as e:
            raise CustomException(e, sys)    

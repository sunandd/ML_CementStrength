import os 
import sys
import pandas as pd
import numpy as np
from src.exception  import CustomException
from  src.logger import logging
from src.utils import save_object

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from dataclasses import dataclass



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()

    #custom function for duplicates replace
    def replace_duplicates_with_mean(X):
        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            unique_values, counts = np.unique(col, return_counts=True)
            duplicate_values = unique_values[counts > 1]

            for dup_val in duplicate_values:
                mask = col == dup_val
                mean_value = np.mean(col[~mask].astype(float))  # Calculate the mean of non-duplicate values
                col[mask] = mean_value

            return X    
    


    
    def get_data_transformer_object(self):
        replace_duplicate = replace_duplicates_with_mean()
        try:
            #define the step for  the preprocessor pipeline
            duplicate_replacement_step = ('duplicate_replacement', FunctionTransformer(replace_duplicate))
            scaler_step = ('scaler' , StandardScaler())

            preprocessor = Pipeline(
                steps=[duplicate_replacement_step, scaler_step]
            )

            return preprocessor    



        except Exception as e:
            raise CustomException(e, sys)    


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = 'Concrete compressive strength(MPa, megapascals)'

            #training dataframe
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            #test dataframe
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            transformed_input_train_feature = preprocessor_obj.fit_transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr =np.c_[input_feature_train_df,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_df,np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            logging.info('preprocessor pickel file saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            logging.info('Exception occured in the initiate_datatransformation')
            raise CustomException(e,sys)


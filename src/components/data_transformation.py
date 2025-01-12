# purpose- perform feature engineering, data cleaning, data transformation

# aim - take outputs from data ingestion
# read 
# apply transformation for categorical and numerical features

import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
# imputer handles missing , null values
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

# gives path and inputs needed for data transformation
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifact','preprocessor.pkl')

# for giving input
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    # to create pickle file to convert categorical to numerical
    # perform standard scaling
    def get_data_transformer_object(self):
        '''
        This funciton is resposible for data transformation
        '''

        try:
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns = [
                'gender', 
                'race_ethnicity', 
                'parental_level_of_education',
                'lunch', 
                'test_preparation_course'
                ]
            
            # create pipe line
            num_pipeline = Pipeline(

                # for fit_transform dataset
                # transform for test dataset
                # imputer handles missing values and 
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('Scaler',StandardScaler())
                ]
                    
                )
            
            cat_pipeline=Pipeline(

                # handle missing values
                # converting to numerical values

                steps=[
                    ('imputer', SimpleImputer(strategy="most_frequent")),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )


            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            
            # Column Transformer - combine numerical and categorical pipeline
            preprocessor = ColumnTransformer(
                [
                    # for numerical columns = reading_score, writing_score
                    # inside num_pipeline = num_pipeline
                    # name of pipeline = 'numerical_pipeline  
                    ('numerical_pipeline',num_pipeline,numerical_columns),

                    # for categorical features : 
                    # ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
                    # inside catpipeline
                    # name of pipeline = 'categorical_pipeline 
                    ('categorical_pipeline',cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            # read dataset
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            # log info
            logging.info('Read train and test completed')

            logging.info('obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'
            numerical_columns = ["writing_score","reading_score"]
            
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f'Applying preprocessing object on trainig dataframe and testing dataframe.')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)

            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # np.c_ is used to combine the features and target columns into a single array, 
            # which is a common step before model training or further data processing.
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            # saving preprocessor.pkl file

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )


            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)
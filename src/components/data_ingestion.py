import os
import sys
# for custom exception
from src.exception import CustomException
from src.logger import logging

import pandas as pd

# used to create class variables
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


## components of data ingestion
# 1. read the data
# 2. convert to raw data path
# 3. convert in csv file
# 4. do train test split
# 5. save train and test file
# 6. return train and test data path for data transformation


# decorator - allows us to define class variable
@dataclass
class DataIngestionConfig:
     
    # in data ingestion component any input that is required 
    # will be given through this class
    # to define class variable we use init but using 
    # data class we can derectly define class variable
    train_data_path: str=os.path.join('artifact','train.csv')
    test_data_path: str=os.path.join('artifact','test.csv')
    raw_data_path: str=os.path.join('artifact','data.csv')
    # output - save all the files in this path


class DataIngestion:
    def __init__(self):

        # consists of 3 path values
        self.ingestion_config=DataIngestionConfig()
        # as soon as this class is called, 3 paths will be saved in path var

    def initiate_data_ingestion(self):

        # if data is stored in some data bases 
        # we will create dbclient in utils.py 
        # purpose read the data

        logging.info("Entered data ingestion method or component")
        
        # error handling
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as df')

            # os.makedirs = create folder with the help of 3 paths
            # os.path.dirname = get dir name wrt specific path 
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)

            logging.info('Train test split initiated')
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False,header=True)

            logging.info('Ingestion of data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)

#initiation
if __name__ == "__main__":
    
    obj = DataIngestion()
    obj.initiate_data_ingestion()




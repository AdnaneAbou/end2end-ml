import sys
import os 
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
        AdaBoostRegressor,  
        GradientBoostingClassifier,
        RandomForestRegressor) 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging 
from src.utils import save_object, evaluate_model
import pickle


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr, test_arr):
        try:
            logging.info("Split training & testing input data")

            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1], 
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]

            ) 

            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decision trees" : DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingClassifier(),
                "Linear Regression" : LinearRegression(),
                "K_Neighbors Regressor" : KNeighborsRegressor(),
                "XGBoost Regressor" : XGBRegressor(),
                "CatBoostRegressor" : CatBoostRegressor(verbose=False),
                "AdaBoostRegressor" : AdaBoostRegressor(),

            }
            model_report:dict=evaluate_model(X_train=X_train , y_train=y_train, X_test=X_test,y_test = y_test , models = models)

            # Best Model score from model_report dictionary

            best_model_score = max(sorted(model_report.values()))

            # Get the Name of the best model score from model_report dictionary 

            best_model_name = list(model_report.keys())[ list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best model score found ')

            logging.info(f"Best model score found on both Training and Test dataset ")


           
            # Use the preprocessor object to transform new data
            # preprocessor_obj = pickle.load('artifacts\preprocessor.pkl')
            # new_data = ...
            # preprocessed_data = preprocessor.transform(new_data)


            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

            
        except Exception as e:
            raise CustomException(e,sys)


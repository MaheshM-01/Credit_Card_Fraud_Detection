

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve
)
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# preprocessing pipeline

def build_preprocessing(cfg: dict):
    # pca features pipeline
    pca_processor= Pipeline(steps=[
        ('imputer',SimpleImputer(strategy=cfg["preprocessing"]['pca_imputer_strategy'])),
        ('pca',PCA(n_components=0.95,random_state=42))
    ])

    # raw features pipeline
    raw_processor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=cfg['preprocessing']['raw_imputer_strategy'])),
        ('scaling',RobustScaler())
    ])

    # combine both pca and raw features using columntransfer
    preprocessor= ColumnTransformer(transformers=[
        ('pca',pca_processor,cfg["preprocessing"]["pca_cols"]),
        ('raw',raw_processor,cfg["preprocessing"]["raw_cols"])

    ])

    return preprocessor

#model pipeline

def model_building(cfg: dict):
    preprocessor= build_preprocessing(cfg)

    if cfg["training"]['estimator']=="xgboost":
        model= XGBClassifier(**cfg["training"]["xgboost_params"])
    else:
        model = RandomForestClassifier(**cfg["training"]["random_forest_params"])

    
    # full pipeline with smote
    steps=[('preprocessing',preprocessor)]
    if cfg['training']['use_smote']:
        steps.append(('smote',SMOTE(random_state=42)))
    steps.append(('classifier',model))

    pipe = ImbPipeline(steps=steps)
    return pipe



#train  for the model
def model_training(pipe,x_train,y_train):
    pipe.fit(x_train,y_train)
    return pipe


# save and load the model

def save_model(pipe,cfg:dict):
    os.makedirs(cfg['paths']['models_dir'],exist_ok=True)
    model_path = os.path.join(cfg['paths']["models_dir"],cfg['serialization']['model_filename'])
    joblib.dump(pipe,model_path)
    print(f"model saved to {model_path}")

def load_model(cfg: dict):
    model_path=os.path.join(cfg['paths']['models_dir'],cfg['serialization']['model_filename'])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f' no model found at {model_path}')
    return joblib.load(model_path)
print('models completed')
#models completed


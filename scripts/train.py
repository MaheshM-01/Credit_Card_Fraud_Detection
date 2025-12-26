
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0,project_root)

import pandas as pd
from sklearn.model_selection import train_test_split

from config import loading_config
from src.data_loading import loading_raw_data
from src.models import build_preprocessing,model_building,model_training,save_model
from src.evaluation import evaluate_model



def main():
    # load configuration
    cfg = loading_config()

    # loading the raw data
    credit = loading_raw_data(cfg)
    print(f'[run_train] data shape : {credit.shape}')

    # split data into train/test
    
    target = cfg['data']['target']
    x=credit.drop(columns=[target])
    y=credit[target]

    #train test split
    x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=cfg['training']['test_size'],random_state=42,stratify=y if cfg['training']['stratify']else None)
    print(f'[run_train] train: {x_train.shape}, test: {x_test.shape}')

    # building and training the model
    pipe=model_building(cfg)
    pipe=model_training(pipe,x_train,y_train)

    # evaluation model (print report and plots)
    results = evaluate_model(pipe, x_test, y_test, threshold=0.98, save_figures=True)

    # save model
    save_model(pipe,cfg)

if __name__ == "__main__":
    main()

    
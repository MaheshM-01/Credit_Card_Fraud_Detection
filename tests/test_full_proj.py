
import sys
import os
import pandas as pd
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0,project_root)

from config import loading_config
from src.data_loading import loading_raw_data
from src.models import model_building,load_model

def test_full_pipeline():
    cfg=loading_config()
    print('\n test config loaded successfully')

    # load raw training data 
    credit=loading_raw_data(cfg)
    assert not credit.empty
    print(f'test processed data shape: {credit.shape}')

    # preprocessing
    credit_sample = credit.sample(5,random_state=42).drop(columns=[cfg['data']['target']])

    #build model pipeline
    pipe = model_building(cfg)
    assert hasattr(pipe,'predict')
    print('test model pipeline build successfully')

    # load trained model
    trained_pipeline= load_model(cfg)
    print('test trained model loaded successfully')

    # predict on sample
    y_pred = trained_pipeline.predict(credit_sample)
    print(f'test prediction: {y_pred}')
    print('test prediction for readable formate:',["Yes" if val == 1 else "No"for val in y_pred],)

    assert y_pred.shape[0]== credit_sample.shape[0]

test_full_pipeline()
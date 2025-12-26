
import os
import sys
import pandas as pd
 
project_roots = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_roots not in sys.path:
    sys.path.insert(0,project_roots)

from config import loading_config
from src.models import load_model

def main():
    cfg= loading_config()

    # load raw new credit data
    new_data_paths = os.path.join(cfg['paths']['data_dir'],'new_credit.csv')
    credit_new_raw = pd.read_csv(new_data_paths)
    print(f'[predict] raw new data shape : {credit_new_raw.shape}')

    # drop target column
    target = cfg['data']['target']
    if target in credit_new_raw.columns:
        credit_new_raw=credit_new_raw.drop(columns=[target])

    # load trained model
    pipe = load_model(cfg)
    print('[predict ] model loaded successfully')

    # predict credit probabilities and classes
    y_proba = pipe.predict_proba(credit_new_raw)[:,1]
    y_pred = pipe.predict(credit_new_raw)

    # prediction
    credit_new_raw['credit_proba']= y_proba
    credit_new_raw['credit_pred']= y_pred

    # readable formate credit column
    credit_new_raw['class prediction']= credit_new_raw['credit_pred'].map({0: "No", 1: "Yes"})

    # save predictionst to the csv file
    output_path = os.path.join (cfg['paths']['models_dir'],'predictions.csv')
    credit_new_raw.to_csv(output_path,index=False) 
    print(f'[predict] predictions saved to { output_path}')

    # sample predictions of unseen dataset
    print(credit_new_raw.head())

if __name__=='__main__':
    main()
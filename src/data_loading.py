
import sys 
import os
 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import loading_config
import pandas as pd

def loading_raw_data(cfg: dict) -> pd.DataFrame:
    '''
    loading the raw credit card fraud detection data set '''

    raw_path= os.path.join(cfg["paths"]["data_raw"],cfg["data"]["raw_file"])
    credit= pd.read_csv(raw_path)
    return credit


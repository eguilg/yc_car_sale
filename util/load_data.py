import pandas as pd
import numpy as np




def load_raw_data(path='../data/yancheng_train.csv'):

    raw_data = pd.read_csv('../data/yancheng_train.csv', na_values=['-'])
    raw_data.sale_date = pd.to_datetime(raw_data.sale_date, format = '%Y%m')

    return raw_data






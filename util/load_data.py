import pandas as pd
import numpy as np




def load_raw_data(base_path='../data/'):
    path = base_path+'yancheng_train.csv'
    raw_data = pd.read_csv(path, na_values=['-'])
    raw_data.sale_date = pd.to_datetime(raw_data.sale_date, format = '%Y%m')

    return raw_data





def load_test_data(base_path='../data/'):
    path = base_path+'yancheng_testA_20171225.csv'
    test_data = pd.read_csv(path, na_values=['-'])
    # raw_data.sale_date = pd.to_datetime(raw_data.sale_date, format = '%Y%m')

    return test_data

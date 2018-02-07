import pandas as pd
import numpy as np

def load_raw_data(path='../data/yancheng_train.csv'):

    parser = lambda date: pd.datetime.strptime(date, '%Y%m')
    train_data = pd.read_csv(path, parse_dates=['sale_date'], date_parser=parser)
    train_data.sort_values(by=['sale_date'], inplace=True)
    return train_data
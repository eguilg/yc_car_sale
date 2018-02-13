import numpy as np
import pandas as pd
from preprocess.preprocess import load_preprocessed_data
from feature.time_series import gen_time_series_features
if __name__ == '__main__':
    data = load_preprocessed_data(base_path='data/',one_hot=True)
    data = gen_time_series_features(data,3,3)
    data.to_csv('data/time_series_data_y3m3.csv',index=False)
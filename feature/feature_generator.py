import pandas as pd
import numpy as np
import os
from preprocess.preprocess import load_preprocessed_data
from util.load_data import load_test_data

def gen_price_diff_feature(data):
    data['price_diff'] = (data.price-data.price_lower)/(data.price_upper-data.price_lower)
    return data


def load_full_feature_data(base_path):
    path = base_path + 'full_feature_data.csv'
    if os.path.exists(path):
        print('loading existing full feature data file..')
        return pd.read_csv(path)
    else:
        print('full feature data dose not exist, start feature generation...')

    data = load_preprocessed_data(base_path)

    data = gen_price_diff_feature(data)

    data.to_csv(path,index=False)
    return data
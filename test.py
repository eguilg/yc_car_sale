import numpy as np
import pandas as pd
from preprocess.preprocess import load_preprocessed_data
from feature.time_series import load_test_time_series, load_train_time_series
if __name__ == '__main__':
    load_test_time_series(lb_year=4,lb_mon=6)
    load_train_time_series(lb_year=4,lb_mon=6)

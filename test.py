import numpy as np
import pandas as pd
from preprocess.preprocess import load_preprocessed_data
from feature.time_series import load_test_time_series, load_train_time_series
if __name__ == '__main__':
    YEAR_SEQ_LEN = 4
    MONTH_SEQ_LEN = 6
    sale_quantity, class_feature_train, year_seq_train, month_seq_train = load_train_time_series(lb_year=YEAR_SEQ_LEN,
                                                                               lb_mon=MONTH_SEQ_LEN)

    class_feature_test, year_seq_test, month_seq_test =load_test_time_series()
    print(class_feature_test.shape,year_seq_test.shape,month_seq_test.shape)
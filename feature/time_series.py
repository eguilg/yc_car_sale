import pandas as pd
import numpy as np
from preprocess.preprocess import load_preprocessed_data




# 生成目标月最近look_back_mon月的历史特征数据
def _gen_time_series_features(preprocessed_class_data, look_back_year = 3,look_back_mon=3,
                              not_equipment_columns = ['class_id_encoded'] + ['brand_id_' + str(k) for k in range(36)]):

    res = pd.DataFrame(preprocessed_class_data[['time_index','sale_quantity']+not_equipment_columns])
    equipment_columns = [col for col in preprocessed_class_data if col not in not_equipment_columns]
    # for item in preprocessed_class_data:

    for i in reversed(range(1, look_back_year + 1)):
        res_y = pd.DataFrame(columns=equipment_columns)
        for item_index in preprocessed_class_data.index:
            t_index = preprocessed_class_data['time_index'] == preprocessed_class_data.loc[item_index,'time_index'] - i*12
            if t_index.sum() == 1:
                res_y.loc[len(res_y), :] = preprocessed_class_data.loc[t_index, equipment_columns].values[0]
            else:
                res_y.loc[len(res_y), :] = [-1 for i in range(len(equipment_columns))]
        res = res.join(res_y, how='outer', lsuffix='', rsuffix='_y' + str(i))

    for i in reversed(range(1, look_back_mon + 1)):
        res_mon = pd.DataFrame(columns=equipment_columns)
        for item_index in preprocessed_class_data.index:
            t_index = preprocessed_class_data['time_index'] == preprocessed_class_data.loc[item_index,'time_index'] - i
            if t_index.sum() == 1:
                res_mon.loc[len(res_mon), :] = preprocessed_class_data.loc[t_index, equipment_columns].values[0]
            else:
                res_mon.loc[len(res_mon), :] = [-1 for i in range(len(equipment_columns))]
        res = res.join(res_mon, how='outer', lsuffix='', rsuffix='_m' + str(i))

    return res


def gen_time_series_features(preprocessed_monthly_data, look_back_year = 3,look_back_mon=3):

    return preprocessed_monthly_data.groupby(by=['class_id'], as_index=False).apply(_gen_time_series_features,
                                                                                    look_back_year,look_back_mon)





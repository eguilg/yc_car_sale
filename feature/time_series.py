import pandas as pd
import numpy as np
import os
from feature.feature_generator import load_full_feature_data
from util.load_data import load_test_data


not_equipment_columns = []#['class_id_encoded'+ str(m) for m in range(140)] + ['brand_id_' + str(m) for m in range(36)]

def __load_class_brand_id_features(base_path='data/',test = True):

    path = base_path+(lambda x:'test' if x==True else 'train')(test)+'_class_features.csv'

    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        preprocessed_data = load_full_feature_data(base_path)
        if test:

            test_data = load_test_data(base_path)
            test_data.predict_date = pd.to_datetime(test_data.predict_date, format='%Y%m')
            test_data['year'] = test_data.predict_date.apply(lambda x: x.year)
            test_data['month'] = test_data.predict_date.apply(lambda x: x.month)
            test_data['time_index'] = (test_data.year - 2012) * 12 + test_data.month

            res = pd.DataFrame(test_data['time_index'])
            class_brand_id = pd.DataFrame(columns=not_equipment_columns)
            for item_index in test_data.index:
                class_brand_id.loc[item_index, :] = preprocessed_data[
                    preprocessed_data['class_id'] == test_data.loc[item_index, 'class_id']].iloc[0][
                    not_equipment_columns]
            res = res.join(class_brand_id, how='outer')
        else:
            res = preprocessed_data[['time_index']+not_equipment_columns]

        res.to_csv(path,index=False)
        return res

def __load_time_series_features(base_path, lb_num = 3,lb_year=True,test=True):

    path = base_path + (lambda x: 'test' if x == True else 'train')(test) + '_lb'\
           +( (lambda x: 'y' if x == True else 'm')(lb_year))+str(lb_num)+'_features.csv'

    if os.path.exists(path):
        return pd.read_csv(path,na_values='-')
    else:
        preprocessed_data = load_full_feature_data(base_path)

        equipment_columns = [col for col in preprocessed_data.columns if
                             col not in not_equipment_columns + ['class_id']]
        index_data = preprocessed_data
        if test:
            test_data = load_test_data(base_path)
            test_data.predict_date = pd.to_datetime(test_data.predict_date, format='%Y%m')
            test_data['year'] = test_data.predict_date.apply(lambda x: x.year)
            test_data['month'] = test_data.predict_date.apply(lambda x: x.month)
            test_data['time_index'] = (test_data.year - 2012) * 12 + test_data.month
            index_data = test_data
        k = 1
        if lb_year:
            k = 12

        res_sq = pd.DataFrame(columns=equipment_columns)
        for item_index in index_data.index:
            t_index_class_id = preprocessed_data['class_id'] == index_data.loc[item_index, 'class_id']
            t_index_time = preprocessed_data['time_index'] == index_data.loc[item_index, 'time_index'] - lb_num*k
            t_index = t_index_class_id * t_index_time
            if t_index.sum() == 1:
                res_sq.loc[item_index, :] = preprocessed_data.loc[t_index, equipment_columns].values[0]
            else:
                res_sq.loc[item_index, :] = [None for i in range(len(equipment_columns))]
        res_sq.to_csv(path,index=False,na_rep='-')
        return res_sq

def load_train_time_series(base_path = 'data/',lb_year=3,lb_mon=3, mon_delay = 1):

    class_brand_id = __load_class_brand_id_features(base_path,test=False)

    lb_year_data = pd.DataFrame()
    for i in reversed(range(1+int(mon_delay/12),lb_year+int(mon_delay/12)+1)):
        lb = __load_time_series_features(base_path,lb_num=i,lb_year=True,test=False)
        lb_year_data = lb_year_data.join(lb,how='outer',rsuffix='_'+str(i))


    lb_mon_data = pd.DataFrame()
    for i in reversed(range(mon_delay, lb_mon + mon_delay)):
        lb = __load_time_series_features(base_path,lb_num=i, lb_year=False, test=False)
        lb_mon_data = lb_mon_data.join(lb, how='outer', rsuffix='_'+str(i))


    y = load_full_feature_data(base_path)['sale_quantity']

    return y,class_brand_id,lb_year_data,lb_mon_data


def load_test_time_series(base_path ='data/',lb_year=3,lb_mon=3,mon_delay = 1):
    class_brand_id = __load_class_brand_id_features(base_path, test=True)

    lb_year_data = pd.DataFrame()
    for i in reversed(range(1+int(mon_delay/12),lb_year+int(mon_delay/12)+1)):
        lb = __load_time_series_features(base_path, lb_num=i, lb_year=True, test=True)
        lb_year_data = lb_year_data.join(lb, how='outer', rsuffix='_' + str(i))

    lb_mon_data = pd.DataFrame()
    for i in reversed(range(mon_delay, lb_mon + mon_delay)):
        lb = __load_time_series_features(base_path, lb_num=i, lb_year=False, test=True)

        lb_mon_data = lb_mon_data.join(lb, how='outer', rsuffix='_' + str(i))


    return class_brand_id,lb_year_data,lb_mon_data
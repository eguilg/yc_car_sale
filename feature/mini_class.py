import pandas as pd
import numpy as np
from preprocess.preprocess import load_preprocessed_data
import os


# 生成配置完全不同的车辆类型细分
def _gen_mini_class(train_data):

    except_columns = ['sale_date','price','sale_quantity','year','month','time_index']
    key_columns =[name for name in train_data.columns if not name in except_columns]

    mini_classes = pd.DataFrame(np.unique(train_data[key_columns], axis=0),columns=key_columns)
    mini_classes['mini_class_id'] = list(range(len(mini_classes)))
    for i in range(len(mini_classes)):
        index = np.array(train_data[key_columns]==mini_classes[key_columns].iloc[i]).all(axis=1)
        train_data.loc[index,'mini_class_id']=mini_classes['mini_class_id'].iloc[i]
        mini_classes.loc[i, 'record_num'] = index.sum()
        mini_classes.loc[i, 'first_time'] = train_data.loc[index, 'time_index'].min()
        mini_classes.loc[i, 'last_time'] = train_data.loc[index, 'time_index'].max()
        # mini_classes.loc[i, 'mean_quantity'] = train_data.loc[index, 'sale_quantity'].mean()
        # mini_classes.loc[i, 'mean_price'] = train_data.loc[index, 'price'].mean()
    return train_data, mini_classes



def load_data_with_mini_classes(data_path = '../data/train_data_with_mini_classes.csv',
                                mini_class_index_path = '../data/mini_class_index.csv'):

    if os.path.exists(data_path) and os.path.exists(mini_class_index_path):
        print('loading existing mini class data files...')
        return pd.read_csv(data_path),pd.read_csv(mini_class_index_path)
    else:
        print('mini class file does not exist, generating...')
        train_data , mini_classes =_gen_mini_class(load_preprocessed_data())

        train_data.to_csv(data_path,index=False)
        mini_classes.to_csv(mini_class_index_path,index=False)
        print('generating done.')
        return  train_data,mini_classes

load_data_with_mini_classes()
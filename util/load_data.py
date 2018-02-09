import pandas as pd
import numpy as np
import datetime

def price_lower(price_level):
    price_level = price_level.replace('WL','')
    price_level = price_level.replace('W', '')
    boundary= price_level.split('-')
    if(len(boundary)==1):
        return 0
    else:
        return int(boundary[0])

def price_upper(price_level):
    price_level = price_level.replace('WL', '')
    price_level = price_level.replace('W', '')
    boundary = price_level.split('-')
    if (len(boundary) == 1):
        return  int(boundary[0])
    else:
        return  int(boundary[1])

def load_raw_data(path='../data/yancheng_train.csv'):

    train_data = pd.read_csv('../data/yancheng_train.csv', na_values=['-'])
    train_data.sale_date = pd.to_datetime(train_data.sale_date, format = '%Y%m')

    train_data['year'] = train_data.sale_date.apply(lambda x:x.year)
    train_data['month'] = train_data.sale_date.apply(lambda x:x.month)
    train_data['time_index'] = (train_data.year-2012)*12+train_data.month
    train_data.level_id.fillna(0, inplace=True)
    train_data.level_id=train_data.level_id.astype(int)
    train_data.loc[train_data.power=='81/70','power']=81
    train_data.power = train_data.power.astype(float)
    train_data.loc[train_data.engine_torque == '155/140','engine_torque'] = 155
    train_data.engine_torque.fillna(100,inplace=True)
    train_data.engine_torque = train_data.engine_torque.astype(float)
    train_data.fuel_type_id.fillna(1, inplace=True)
    train_data.fuel_type_id = train_data.fuel_type_id.astype(int)

    train_data.TR = train_data.TR.map({'0':0,
                                       '1':1,
                                       '4':4,
                                       '5;4':4.5,
                                       '5':5,
                                       '6':6,
                                       '7':7,
                                       '8;7':7.5,
                                       '8':8,
                                       '9':9}).astype(float)
    train_data.gearbox_type = train_data.gearbox_type.map({'AMT':0,
                                                           'AT':1,
                                                           'AT;DCT':2,
                                                           'CVT':3,
                                                           'DCT':4,
                                                           'MT':5,
                                                           'MT;AT':6}).astype(int)
    train_data.if_charging = train_data.if_charging.map({'L':0,'T':1})
    train_data.rated_passenger = train_data.rated_passenger.map({'4':4,
                                                                 '5':5,
                                                                 '4-5':4.5,
                                                                 '5-7':6.1,
                                                                 '5-8':6.6,
                                                                 '6-7':6.5,
                                                                 '7':7,
                                                                 '6-8':7.1,
                                                                 '7-8':7.5,
                                                                 '9':9}).astype(float)
    train_data['price_lower'] = train_data.price_level.apply(price_lower)
    train_data['price_upper'] = train_data.price_level.apply(price_upper)
    train_data.drop('price_level',axis=1,inplace=True)
    train_data.sort_values(by=['class_id', 'time_index'], inplace=True)
    return train_data
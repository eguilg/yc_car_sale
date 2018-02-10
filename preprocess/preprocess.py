from util.load_data import load_raw_data
import pandas as pd
import os

def _price_lower(price_level):
    price_level = price_level.replace('WL','')
    price_level = price_level.replace('W', '')
    bound= price_level.split('-')
    if(len(bound)==1):
        return 0
    else:
        return int(bound[0])

def _price_upper(price_level):
    price_level = price_level.replace('WL', '')
    price_level = price_level.replace('W', '')
    bound = price_level.split('-')
    if (len(bound) == 1):
        return  int(bound[0])
    else:
        return  int(bound[1])


def _data_cleaning(train_data):

    # handle abnormal records
    train_data.loc[train_data.power == '81/70', 'power'] = 75.5
    train_data.power = train_data.power.astype(float)
    train_data.loc[train_data.engine_torque == '155/140', 'engine_torque'] = 147.5

    # handle na
    train_data.level_id.fillna(0, inplace=True)
    train_data.level_id = train_data.level_id.astype(int)
    train_data.engine_torque.fillna(100, inplace=True)
    train_data.engine_torque = train_data.engine_torque.astype(float)
    train_data.fuel_type_id.fillna(1, inplace=True)
    train_data.fuel_type_id = train_data.fuel_type_id.astype(int)



    # categories mapping
    train_data.if_charging = train_data.if_charging.map({'L': 0, 'T': 1})
    train_data.rated_passenger = train_data.rated_passenger.map({'4': 4,
                                                                 '5': 5,
                                                                 '4-5': 4.5,
                                                                 '5-7': 6.1,
                                                                 '5-8': 6.6,
                                                                 '6-7': 6.5,
                                                                 '7': 7,
                                                                 '6-8': 7.1,
                                                                 '7-8': 7.5,
                                                                 '9': 9}).astype(float)
    train_data.gearbox_type = train_data.gearbox_type.map({'AMT': 0,
                                                           'AT': 1,
                                                           'AT;DCT': 2,
                                                           'CVT': 3,
                                                           'DCT': 4,
                                                           'MT': 5,
                                                           'MT;AT': 6}).astype(int)
    train_data.TR = train_data.TR.map({'0': 0,
                                       '1': 1,
                                       '4': 4,
                                       '5;4': 4.5,
                                       '5': 5,
                                       '6': 6,
                                       '7': 7,
                                       '8;7': 7.5,
                                       '8': 8,
                                       '9': 9}).astype(float)

    # transform price level to lower and upper bound
    train_data['price_lower'] = train_data.price_level.apply(_price_lower)
    train_data['price_upper'] = train_data.price_level.apply(_price_upper)

    # time indexing
    train_data['year'] = train_data.sale_date.apply(lambda x: x.year)
    train_data['month'] = train_data.sale_date.apply(lambda x: x.month)
    train_data['time_index'] = (train_data.year - 2012) * 12 + train_data.month

    train_data.drop('price_level', axis=1, inplace=True)
    train_data.sort_values(by=['class_id', 'time_index'], inplace=True)

    return train_data



def load_preprocessed_data(path='../data/yancheng_train_cleaned.csv'):
    if os.path.exists(path):
        print('loading existing cleaned data file..')
        return pd.read_csv(path)
    else:
        print('cleaned data dose not exist ,start data cleaning...')
        cleaned_data = _data_cleaning(load_raw_data())
        cleaned_data.to_csv(path,index=False)
        print('data cleaning done, result saved.')
        return cleaned_data
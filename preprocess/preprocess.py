import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
import xgboost as xgb
from util.load_data import load_raw_data

# 该模块包含所有数据清洗、预处理操作，
# 并在最后用 load_preprocessed_data() 函数综合起来


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
    # train_data.if_charging = train_data.if_charging.map({'L': 0, 'T': 1})
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
    # train_data.gearbox_type = train_data.gearbox_type.map({'AMT': 0,
    #                                                        'AT': 1,
    #                                                        'AT;DCT': 2,
    #                                                        'CVT': 3,
    #                                                        'DCT': 4,
    #                                                        'MT': 5,
    #                                                        'MT;AT': 6}).astype(int)
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

    train_data['brand_id'] = LabelEncoder().fit_transform(np.reshape(train_data['brand_id'].values, (-1, 1)))
    train_data['gearbox_type'] = LabelEncoder().fit_transform(np.reshape(train_data['gearbox_type'].values, (-1, 1)))
    train_data['if_charging'] = LabelEncoder().fit_transform(np.reshape(train_data['if_charging'].values, (-1, 1)))
    train_data['class_id_encoded'] = LabelEncoder().fit_transform(np.reshape(train_data['class_id'].values, (-1, 1)))
    # transform price level to lower and upper bounds
    train_data['price_lower'] = train_data.price_level.apply(_price_lower)
    train_data['price_upper'] = train_data.price_level.apply(_price_upper)

    # time indexing
    train_data['year'] = train_data.sale_date.apply(lambda x: x.year)
    train_data['month'] = train_data.sale_date.apply(lambda x: x.month)
    train_data['time_index'] = (train_data.year - 2012) * 12 + train_data.month

    # drop掉旧的表示方式
    train_data.drop('sale_date', axis=1, inplace=True)
    train_data.drop('price_level', axis=1, inplace=True)

    # 排序
    train_data.sort_values(by=['class_id', 'time_index'], inplace=True)

    return train_data

# convert categorical feature to one hot
def one_hot_encode(train_data,categorical_columns):
    for col in categorical_columns:
        if col in train_data.columns:
            enc = OneHotEncoder()
            one_hot = enc.fit_transform(np.reshape(train_data[col].values,(-1,1))).toarray()
            for i in range(one_hot.shape[1]):
                train_data[col+'_'+str(i)] = one_hot[:,i]
            train_data.drop([col],axis=1,inplace=True)
    return train_data


def _scorer(ground_truth, pred):
    # return mean_squared_error(np.square(ground_truth), np.square(pred))
    return mean_squared_error(np.expm1(ground_truth), np.expm1(pred))
# using XGBRegressor to predict the missing price data
# note to input the preprocessed data with one hot encode
def predict_missing_price(preprocessed_data, one_hot=False):

    test_index = preprocessed_data.price != preprocessed_data.price

    feature_columns = [i for i in preprocessed_data.columns if i not in ['class_id','price']]
    y_column = ['price']
    testX =preprocessed_data.loc[test_index, feature_columns].values

    trainX = preprocessed_data.loc[(1-test_index).astype(bool), feature_columns].values
    trainY = preprocessed_data.loc[(1-test_index).astype(bool), y_column].values

    # plt.hist(trainY)
    # plt.show()

    # 销量数据使用log1p处理后更接近正态分布，比sqrt处理要好
    # trs = FunctionTransformer(func=np.sqrt, inverse_func=np.square)
    trs = FunctionTransformer(func=np.log1p,inverse_func=np.expm1)
    scaler = MinMaxScaler()
    trainX = scaler.fit_transform(trainX)
    trainY = trs.fit_transform(np.reshape(trainY,(-1,1)))

    # plt.hist(trainY)
    # plt.show()
    print(trainX.shape,trainY.shape)
    clf = xgb.XGBRegressor(seed=12)

    if one_hot:
        # ONE HOT with norm PARAMS sqare
        grid = [{
            'booster': ['gbtree'],
            'learning_rate': [0.1],
            # 'min_child_weight':[],
            'max_depth': [2],
            'gamma': [1],
            'subsample': [0.3],
            'colsample_bytree': [0.3],
            'reg_alpha': [1.0],
            'reg_lambda': [0.85],
            'scale_pos_weight': [1]
        },
        ]
    else:
        # no one hot PARAMS sqrt

        # grid = [{
        #     'booster': ['gbtree'],
        #     'learning_rate': [0.1],
        #     # 'min_child_weight':[],
        #     'max_depth': [2],
        #     'gamma': [0.7],
        #     'subsample': [0.1],
        #     'colsample_bytree': [0.3],
        #     'reg_alpha': [0.5],
        #     'reg_lambda': [0.3],
        #     'scale_pos_weight': [1]
        # },
        # ]

        # no one hot PARAMS log1p
        grid = [{
            'booster': ['gbtree'],
            'learning_rate': [0.25],
            # 'min_child_weight':[],
            'max_depth': [2],
            'gamma': [0.09],
            'subsample': [0.1],
            'colsample_bytree': [0.95],
            'reg_alpha': [0.5],
            'reg_lambda': [0.25],
            'scale_pos_weight': [1]
        },
        ]

    gridCV = GridSearchCV(estimator=clf, param_grid=grid,
                          scoring= make_scorer(_scorer,greater_is_better=False),
                          iid=False, n_jobs=-1, cv=6, verbose=1)


    gridCV.fit(trainX, trainY)

    print("best params:", gridCV.best_params_)
    print('best score:', gridCV.best_score_)
    testX = scaler.transform(testX)
    predY = np.reshape(gridCV.predict(testX),(-1,1))
    preprocessed_data.loc[test_index, y_column] = trs.inverse_transform(predY)


    return preprocessed_data

def _gen_data_per_classmonth(df_per_classmonth):
    res = pd.Series(index=df_per_classmonth.columns)

    # 对于每月各车型细分的销售数量求和
    res['sale_quantity'] = df_per_classmonth['sale_quantity'].sum()

    # 其他特征求对销售数量的加权平均
    ave_col = [col for col in df_per_classmonth.columns if col != 'sale_quantity']
    res[ave_col] = np.average(df_per_classmonth[ave_col],
                              weights=df_per_classmonth['sale_quantity'],
                              axis=0)

    return res


def load_preprocessed_data(path='../data/yancheng_train_preprocessed.csv',
                           one_hot = True):
    if one_hot:
        path =  path.split('.csv')[0]+'_onehot.csv'

    if os.path.exists(path):
        print('loading existing preprocessed data file..')
        return pd.read_csv(path)
    else:
        print('preprocessed data dose not exist ,start data processing...')
        # data cleaning
        data = _data_cleaning(load_raw_data())


        # fix misssing price
        data = predict_missing_price(data,one_hot=False)

        # one hot encode
        if one_hot:
            categorical_columns = ['brand_id', 'type_id', 'level_id', 'department_id', 'TR', 'gearbox_type',
                                   'if_charging', 'driven_type_id', 'fuel_type_id', 'newenergy_type_id',
                                   'emission_standards_id', 'if_MPV_id', 'if_luxurious_id']

            data = one_hot_encode(data, categorical_columns)

        # 生成每个class_id 按月的销售记录(根据销量加权平均)
        data = data.groupby(by=['class_id','time_index'],as_index=False).apply(_gen_data_per_classmonth)

        data.to_csv(path,index=False)

        print('data preprocess done, result saved.')
        return data

if __name__ == '__main__':
    # load_preprocessed_data(one_hot=False)
    load_preprocessed_data(one_hot=False)
import pandas as pd
import numpy as np
from util.load_data import load_raw_data

def build_mini_class():
    train_data = load_raw_data()
    # train_data.sort_values(by=['class_id','time_index'], inplace=True)
    classes = np.unique(train_data.class_id)
    except_columns = ['sale_date','price','sale_quantity','year','month','time_index']
    key_columns =[name for name in train_data.columns if not name in except_columns]
    mini_classes = pd.DataFrame(columns=['mini_class_id']+key_columns)
    # print(key_columns)
    print(len(np.unique(train_data[key_columns],axis=0)))
    return mini_classes

#
# train_data = load_raw_data()
# for name in train_data.columns:
#     n = len(np.unique(train_data[name]))
#     # if n <=20:
#     print(n,name,train_data[name].dtype,np.unique(train_data[name]))
build_mini_class()
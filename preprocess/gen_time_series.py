import pandas as pd
import numpy as np
from util.load_data import load_raw_data

def gen_sale_quantity_series(look_back=3):

    raw_train = load_raw_data()
    sale_quantity_perclass = raw_train[['sale_date', 'class_id', 'sale_quantity']]\
        .groupby(by=['class_id','sale_date'], as_index=False).sum()
    sale_quantity_perclass.sort_values(by=['class_id','sale_date'])
    classes, class_sale_counts = np.unique(sale_quantity_perclass['class_id'], return_counts=True)
    testset = pd.DataFrame(columns=list(range(1,look_back+1))+['class_id'])
    dataset = pd.DataFrame(columns=list(range(1,look_back+1))+['y','class_id'])
    for id in classes:
        sale_quantity = sale_quantity_perclass[sale_quantity_perclass.class_id == id]
        x_test = []
        if sale_quantity.shape[0]>look_back: # 销售记录数量大于观察窗口长度
            for i in range(sale_quantity.shape[0]-look_back):
                x = sale_quantity.sale_quantity.iloc[i:i+look_back].values.tolist()
                y = sale_quantity.sale_quantity.iloc[i+look_back]
                dataset.loc[dataset.shape[0]] = x+[y,id]
            x_test = sale_quantity.sale_quantity.iloc[-look_back:].values.tolist()

        else: # 销售记录数量小于等于观察窗口长度
            y = sale_quantity.sale_quantity.iloc[-1]
            x = sale_quantity.sale_quantity.iloc[:-1].values.tolist()
            # 前面缺的值用均值补
            x = [np.mean(x) for i in range(look_back-len(x))]+x
            dataset.loc[dataset.shape[0]] = x + [y,id]
            x_test = sale_quantity.sale_quantity.values.tolist()
            x_test = [np.mean(x_test) for i in range(look_back-len(x_test))]+x_test
        testset.loc[testset.shape[0]] = x_test + [id]
    return dataset, testset

if __name__ == "__main__":
    look_back = 12
    dataset ,testset= gen_sale_quantity_series(look_back=look_back)
    dataset.to_csv('../data/train_sale_lb='+str(look_back)+'.csv',index=False)
    testset.to_csv('../data/test_sale_lb=' + str(look_back) + '.csv', index=False)
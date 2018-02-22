import numpy as np
import pandas as pd
import datetime
import keras
from keras.models import Model
from keras.layers import Input,Dense,GRU,LSTM,Masking,Merge,Dropout
from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau,EarlyStopping,CSVLogger
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.metrics import mean_squared_error

from feature.time_series import load_train_time_series, load_test_time_series
from util.load_data import load_test_data

def rmse_loss(ground_truth, pred):
    # return mean_squared_error(ground_truth, pred)
    return K.sqrt(K.mean(K.square(ground_truth-pred),axis=-1))

def create_model(year_seq_shape,month_seq_shape,
                 seq_size = 100, final_dense_size = 32,
                 lr = 0.1, decay = 1e-5):



    year_seq_input = Input(shape=year_seq_shape, name='year_seq_input')
    year_seq_mask = Masking(mask_value=-1, input_shape=year_seq_shape,
                            name='year_seq_mask')(year_seq_input)
    year_seq_out = GRU(seq_size, input_shape=year_seq_shape,
                        dropout_W=0.1, dropout_U=0.1, name='year_seq')(year_seq_mask)

    month_seq_input = Input(shape=month_seq_shape, name='month_seq_input')
    month_seq_mask = Masking(mask_value=-1, input_shape=month_seq_shape,
                             name='month_seq_mask')(month_seq_input)
    month_seq_out = GRU(seq_size, input_shape=month_seq_shape,
                         dropout_W=0.1, dropout_U=0.1, name='month_seq')(month_seq_mask)

    seq_merge = Merge(name='seq_merge', mode = 'ave')([year_seq_out, month_seq_out])

    dense_after_merge = Dense(seq_size, name = 'dense_after_merge')(seq_merge)
    drop_out = Dropout(0.1, name = 'drop_out')(dense_after_merge)
    final_dense = Dense(final_dense_size,name='final_dense')(drop_out)
    main_out = Dense(1,name='main_out',activation='softplus')(final_dense)

    model = Model(inputs=[ year_seq_input, month_seq_input],
                  outputs=[main_out])

    adam = Adam()
    model.compile(optimizer=adam, loss=rmse_loss)
    model.summary()

    return model



def cross_validation(sample_num, cv, seed):

    np.random.seed(seed)
    folds = []
    all = list(range(sample_num))
    remains = all
    for i in range(cv):
        ret = {}
        ret['vali'] = np.random.choice(remains,int(sample_num/cv))
        ret['train'] = list(set(all)^set(ret['vali']))
        remains = list(set(remains)^set(ret['vali']))

        folds.append(ret)

    return folds

if __name__ == '__main__':

    YEAR_SEQ_LEN = 5
    MONTH_SEQ_LEN = 24

    NUM_EPOCH = 300
    BATCH_SIZE = 50
    LR = 0.1
    DECAY = 1e-5
    CV = 6

    timestamp = datetime.datetime.now().strftime('%m%d%H%M')
    model_name = 'lstm_y' + str(YEAR_SEQ_LEN) + 'm' + str(MONTH_SEQ_LEN) + '_' + 'e' \
                 + str(NUM_EPOCH) + 'b' + str(BATCH_SIZE) + '_' + timestamp

    sale_quantity, class_feature_train, year_seq_train, month_seq_train = load_train_time_series(lb_year=YEAR_SEQ_LEN,
                                                                                                 lb_mon=MONTH_SEQ_LEN)

    class_feature_test, year_seq_test, month_seq_test =load_test_time_series(lb_year=YEAR_SEQ_LEN,
                                                                             lb_mon=MONTH_SEQ_LEN)

    Y_all = sale_quantity
    X1_all = class_feature_train
    X2_all = year_seq_train
    X3_all = month_seq_train

    # X1_all_filter = class_feature_train != class_feature_train
    X2_all_filter = year_seq_train != year_seq_train
    X3_all_filter = month_seq_train != month_seq_train

    X2_all_min = X2_all.min(axis=0)
    X2_all_min.fillna(-1,inplace =True)
    X2_all.fillna(X2_all_min,inplace=True)
    X3_all_min = X3_all.min(axis=0)
    X3_all_min.fillna(-1, inplace=True)
    X3_all.fillna(X3_all_min, inplace = True)
    print(X1_all.shape,X2_all.shape,X3_all.shape)
    X1_test = class_feature_test
    X2_test = year_seq_test
    X3_test = month_seq_test

    X2_test_filter = year_seq_test != year_seq_test
    X3_test_filter = month_seq_test != month_seq_test

    X2_test_min = X2_test.min(axis=0)
    X2_test_min.fillna(-1, inplace=True)
    X2_test.fillna(X2_test_min,inplace=True)
    X3_test_min = X3_test.min(axis=0)
    X3_test_min.fillna(-1,inplace=True)
    X3_test.fillna(X3_test_min,inplace=True)
    print(X1_test.shape, X2_test.shape, X3_test.shape)

    # normalize the dataset
    scalerX1 = MinMaxScaler(feature_range=(0, 1))
    scalerX2 = MinMaxScaler(feature_range=(0, 1))
    scalerX3 = MinMaxScaler(feature_range=(0, 1))


    X1_all = scalerX1.fit_transform(X1_all)
    X2_all = scalerX2.fit_transform(X2_all)
    X3_all = scalerX3.fit_transform(X3_all)
    Y_all = np.reshape(Y_all,(-1,1))



    X1_test = scalerX1.transform(X1_test)
    X2_test = scalerX2.transform(X2_test)
    X3_test = scalerX3.transform(X3_test)

    X2_all[X2_all_filter] = -1
    X3_all[X3_all_filter] = -1
    X2_test[X2_test_filter] = -1
    X3_test[X3_test_filter] = -1

    X2_all_filter = None
    X3_all_filter = None
    X2_test_filter = None
    X3_test_filter = None

    # reshape input to be [samples, time steps, features]
    X2_all = np.reshape(X2_all, (X2_all.shape[0], YEAR_SEQ_LEN,int(X2_all.shape[1] / YEAR_SEQ_LEN)))
    X3_all = np.reshape(X3_all, (X3_all.shape[0],MONTH_SEQ_LEN,int(X3_all.shape[1] / MONTH_SEQ_LEN)))

    X2_test = np.reshape(X2_test, (X2_test.shape[0], YEAR_SEQ_LEN, int(X2_test.shape[1] / YEAR_SEQ_LEN)))
    X3_test = np.reshape(X3_test, (X3_test.shape[0], MONTH_SEQ_LEN, int(X3_test.shape[1] / MONTH_SEQ_LEN)))
    # create and fit the LSTM network

    folds = cross_validation(X1_all.shape[0],CV,seed=12)
    train_scores = []
    vali_scores = []
    test_predicts = []
    for i in range(len(folds)):

        train_index = folds[i]['train']
        vali_index = folds[i]['vali']

        # X1_train = X1_all[train_index]
        X2_train = X2_all[train_index]
        X3_train = X3_all[train_index]
        Y_train = Y_all[train_index]

        # X1_vali = X1_all[vali_index]
        X2_vali = X2_all[vali_index]
        X3_vali = X3_all[vali_index]
        Y_vali = Y_all[vali_index]

        model = create_model(year_seq_shape=(X2_train.shape[1],X2_train.shape[2]),
                             month_seq_shape=(X3_train.shape[1],X3_train.shape[2]),
                             lr=LR,decay=DECAY)

        logger = CSVLogger('log/'+model_name+'_cv'+str(i)+'.csv')
        earlystop = EarlyStopping(monitor='val_loss', patience=60, verbose=1, min_delta=0.5)
        reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.90, patience=4,verbose=1)
        history = model.fit([X2_train,X3_train], [Y_train],
                            validation_data=([X2_vali,X3_vali],[Y_vali]),
                            callbacks=[earlystop, reduce, logger],
                            epochs=NUM_EPOCH, batch_size=BATCH_SIZE, shuffle=True, verbose=1)




        trainScore = history.history['loss'][-1]
        valiScore = history.history['val_loss'][-1]
        cv_testPredict = model.predict([X2_test, X3_test])
        train_scores.append(trainScore)
        vali_scores.append(valiScore)
        test_predicts.append(cv_testPredict)

        print('Fold: %d' % (i))
        print('Train Score: %.4f rmse' % (trainScore))
        print('Vali Score: %.4f rmse' % (valiScore))

    mean_trainScore = np.mean(train_scores)
    mean_valiScore = np.mean(vali_scores)
    print('Mean train score: %.4f rmse' %(mean_trainScore))
    print('Mean vali score: %.4f rmse' % (mean_valiScore))

    cv_weights = np.average([train_scores,vali_scores],
                         weights=[CV-1,1],axis=0)
    mean_test_predict = np.average(test_predicts,axis=0, weights=1/cv_weights)

    # totallY = np.vstack((trainPredict,valiPredict))
    # inversedY = scalerY.inverse_transform(totallY)
    #
    # trainPredict,valiPredict = inversedY[0:train_size,:], inversedY[train_size:len(inversedY),:]
    # Y = scalerY.inverse_transform(Y)
    #
    # trainY, valiY = Y[0:train_size,:], Y[train_size:len(Y),:]
    # trainScore = math.sqrt(mean_squared_error(trainY[:], trainPredict[:,0]))
    # print('Train Score: %.2f rmse' % (trainScore))
    # valiScore = math.sqrt(mean_squared_error(valiY[:], valiPredict[:,0]))
    # print('vali Score: %.2f rmse' % (valiScore))
    # shift train predictions for plotting
    # fig = plt.figure()
    # ax1=fig.add_subplot(211)
    # ax1.plot(trainPredict,label='pred')
    # ax1.plot(trainY,label='true',alpha =0.5)
    # ax1.legend(['pred','true'])
    # ax1.set_title('train')
    # ax2 = fig.add_subplot(212)
    # ax2.set_title('vali')
    # ax2.plot(valiPredict,label = 'pred')
    # ax2.plot(valiY, label = 'true',alpha =0.5)
    # ax2.legend(['pred','true'])
    #
    # plt.show()


    sub = load_test_data(base_path='data/')
    sub.predict_quantity = np.reshape(mean_test_predict,(mean_test_predict.shape[0]))

    sub.to_csv('sub/'+model_name+'_cv'+str(int(mean_valiScore))+'.csv',index=False)

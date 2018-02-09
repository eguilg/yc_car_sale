import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import datetime
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_squared_error
from preprocess.gen_time_series import gen_sale_quantity_series
# convert an array of values into a dataset matrix
seq_len = 12

epoch = 500
batch_size = 300
dataset ,testset = gen_sale_quantity_series(seq_len=seq_len)
# fix random seed for reproducibility
np.random.seed(7)
# load the dataset
# dataframe = read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataset.values
np.random.shuffle(dataset)
dataset = dataset.astype('float32')
X = dataset[:,:seq_len]
Y = dataset[:,-2]
# normalize the dataset
scalerX = MinMaxScaler(feature_range=(0, 1))
scalerY = MinMaxScaler(feature_range=(0, 1))

X = scalerX.fit_transform(X)
Y = scalerY.fit_transform(np.reshape(Y,(-1,1)))
# split into train and vali sets
train_size = int(len(dataset) * 0.80)
vali_size = len(dataset) - train_size
trainX, valiX = X[0:train_size,:], X[train_size:len(X),:]
trainY, valiY = Y[0:train_size,:], Y[train_size:len(Y),:]


# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], seq_len,1))
valiX = np.reshape(valiX, (valiX.shape[0], seq_len,1))
# create and fit the LSTM network

seq_input = Input(shape=(trainX.shape[1],trainX.shape[2]), dtype='float32', name='seq_input')
lstm_out = LSTM(32,input_shape=(trainX.shape[1],trainX.shape[2]),
               dropout_W=0.2, dropout_U=0.2)(seq_input)
main_out = Dense(trainY.shape[1],name='main_out')(lstm_out)

model = Model(inputs=[seq_input ], outputs=[main_out])

model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()
model.fit([trainX], [trainY], epochs=epoch, batch_size=batch_size, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
valiPredict = model.predict(valiX)

totallY = np.vstack((trainPredict,valiPredict))
inversedY = scalerY.inverse_transform(totallY)

trainPredict,valiPredict = inversedY[0:train_size,:], inversedY[train_size:len(inversedY),:]
Y = scalerY.inverse_transform(Y)

trainY, valiY = Y[0:train_size,:], Y[train_size:len(Y),:]
trainScore = math.sqrt(mean_squared_error(trainY[:], trainPredict[:,0]))
print('Train Score: %.2f rmse' % (trainScore))
valiScore = math.sqrt(mean_squared_error(valiY[:], valiPredict[:,0]))
print('vali Score: %.2f rmse' % (valiScore))
# shift train predictions for plotting
fig = plt.figure()
ax1=fig.add_subplot(211)
ax1.plot(trainPredict,label='pred')
ax1.plot(trainY,label='true',alpha =0.5)
ax1.legend(['pred','true'])
ax1.set_title('train')
ax2 = fig.add_subplot(212)
ax2.set_title('vali')
ax2.plot(valiPredict,label = 'pred')
ax2.plot(valiY, label = 'true',alpha =0.5)
ax2.legend(['pred','true'])

plt.show()


testX = testset.values[:,:seq_len]
testX = np.reshape(scalerX.transform(testX),(len(testX),seq_len,1))

testPredict = model.predict(testX)
testPredict = scalerY.inverse_transform(testPredict)

sub = pd.read_csv('../data/yancheng_testA_20171225.csv')
sub.predict_quantity = np.reshape(testPredict,(testPredict.shape[0]))
timestamp = datetime.datetime.now().strftime('%m%d%H%M')
sub.to_csv('../sub/lstm_sl'+str(seq_len)+'_'+'e'+str(epoch)+'b'+str(batch_size)+'_'+timestamp+'.csv',index=False)
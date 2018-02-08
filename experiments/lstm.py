import numpy
import pandas as pd
import matplotlib.pyplot as plt
import math
import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from preprocess.gen_time_series import gen_sale_quantity_series
# convert an array of values into a dataset matrix
look_back = 1
k = 1.12
dataset ,testset = gen_sale_quantity_series(look_back=look_back)
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
# dataframe = read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataset.values
numpy.random.shuffle(dataset)
dataset = dataset.astype('float32')
X = dataset[:,:look_back]
Y = dataset[:,:-2]
# normalize the dataset
scalerX = MinMaxScaler(feature_range=(0, 1))
scalerY = MinMaxScaler(feature_range=(0, 1))
X = scalerX.fit_transform(X)
Y = scalerY.fit_transform(Y)
# split into train and vali sets
train_size = int(len(dataset) * 0.67)
vali_size = len(dataset) - train_size
trainX, valiX = X[0:train_size,:], X[train_size:len(X),:]
trainY, valiY = Y[0:train_size,:], Y[train_size:len(Y),:]





# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], look_back,1))
valiX = numpy.reshape(valiX, (valiX.shape[0], look_back,1))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(10,input_dim=1,input_length=look_back,dropout_W=0.2, dropout_U=0.2))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()
model.fit(trainX, trainY, epochs=200, batch_size=200, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
valiPredict = model.predict(valiX)
# invert predictions
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([trainY])
# valiPredict = scaler.inverse_transform(valiPredict)
# valiY = scaler.inverse_transform([valiY])
# calculate root mean squared error
totallY = numpy.vstack((trainPredict,valiPredict))
inversedY = k*scalerY.inverse_transform(totallY)
trainPredict,valiPredict = inversedY[0:train_size,:], inversedY[train_size:len(inversedY),:]
Y = scalerY.inverse_transform(Y)
trainY, valiY = Y[0:train_size,:], Y[train_size:len(Y),:]
trainScore = math.sqrt(mean_squared_error(trainY[:], trainPredict[:,0]))
print('Train Score: %.2f mse' % (trainScore))
valiScore = math.sqrt(mean_squared_error(valiY[:], valiPredict[:,0]))
print('vali Score: %.2f mse' % (valiScore))
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


testX = testset.values[:,:look_back]
testX = numpy.reshape(scalerX.transform(testX),(len(testX),look_back,1))

testPredict = model.predict(testX)
testPredict = k*scalerY.inverse_transform(testPredict)
sub = pd.read_csv('../data/yancheng_testA_20171225.csv')
sub.predict_quantity = numpy.reshape(testPredict,(testPredict.shape[0]))
timestamp = datetime.datetime.now().strftime('%m%d%H%M')
sub.to_csv('../sub/lstm_lb'+str(look_back)+'_'+timestamp+'.csv',index=False)
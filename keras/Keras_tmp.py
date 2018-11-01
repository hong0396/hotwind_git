import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sqlalchemy import create_engine
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.linear_model import LassoCV 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from  sklearn.metrics  import  average_precision_score 
from sklearn.metrics import r2_score  
import csv
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# from keras.models import load_model

# model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
# del model  # deletes the existing model

# # returns a compiled model
# # identical to the previous one
# model = load_model('my_model.h5')
 
# load the dataset
dataframe = pd.read_csv('C:\\Users\\guihong\\Documents\\Git_hotwind\\stock\\fix_yahoo_finance/aapl.csv')
dataset = dataframe['y'].values.tolist()
# dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# plt.scatter(X[:200], Y[:200])
# plt.show()

#begin trainning

# SPLIT=0.8*max_length
# X_train, Y_train = X[:SPLIT], Y[:SPLIT]  
# X_test, Y_test = X[SPLIT:], Y[SPLIT:]    


# # single dense model

# model = Sequential()
# model.add(Dense(input_dim=1, units=1))
# model.compile(loss='mse', optimizer='sgd')


# print('Training -----------')
# for step in range(10000):
#     cost =model.train_on_batch(X_train.values, Y_train.values)
#     if step % 50 == 0:
#         print("After %d trainings,the cost: %f" % (step, cost))

 

# # testing
# print('\nTesting ------------')
# cost = model.evaluate(X_test.values, Y_test.values, batch_size=40)
# print('test cost:', cost)
# W, b = model.layers[0].get_weights()
# print('Weights=', W, '\nbiases=', b)

 

# # predict
# Y_pred = model.predict(X_test.values)
# plt.scatter(X_test.values, Y_test.values)
# plt.plot(X_test.values, Y_pred,'r')
# plt.show()






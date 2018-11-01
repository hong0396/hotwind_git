import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sqlalchemy import create_engine
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from  sklearn.metrics  import  average_precision_score 
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoCV
from xgboost import XGBRegressor
from sklearn.linear_model import RidgeCV
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Activation
from keras.layers.core import Dense, Dropout,Activation
from sklearn import preprocessing

train=pd.read_csv('train.csv', index_col=0)
test_df=pd.read_csv('test.csv', index_col=0)


train_len = len(train)
# print(train_len)
# print(train.columns.size)


y=train['SalePrice']
X_df=train.drop(['SalePrice'], axis = 1)


all_df = pd.concat((X_df, test_df), axis=0)

#print(X['MSSubClass'].dtypes)
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)
#pd.get_dummies(X['MSSubClass'], prefix='MSSubClass')
all_df= pd.get_dummies(all_df)
print(all_df.isnull().sum().sum())
all_df = all_df.fillna(all_df.mean())
print(all_df.isnull().sum().sum())



X = all_df.loc[X_df.index]
test_df = all_df.loc[test_df.index]

print(X.columns.size)

X=X.values
test=test_df.values

y = np.log1p(y)
y = y.values



X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2)


# single dense model

model = Sequential()

# model.add(LSTM(200, init='normal',activation='relu',input_dim=303))
# # model.add(Dropout(0.5))
# # model.add(Activation('linear'))
# # model.add(Dense(100, activation='relu'))
# # model.add(Dropout(0.5))
# # model.add(Dense(50, activation='relu'))
# model.add(LSTM(1, activation='relu'))




model.add(Dense(200, init='normal',activation='softmax',input_dim=304))
model.add(Dropout(0.5))
# model.add(Activation('linear'))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='relu'))
# model.add(Activation('tanh'))
# model.add(Activation('softmax')) 
# model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.00001, momentum=0.9, nesterov=True))
 
model.compile(loss='mean_absolute_error', optimizer=SGD(lr=0.0000001, momentum=0.95, nesterov=True))
# model.compile(loss='mean_squared_error', optimizer='rmsprop')
# print('Training -----------')
# for step in range(10000):
#     cost =model.train_on_batch(X_train, Y_train)
#     if step % 50 == 0:
#          print("After %d trainings,the cost: %f" % (step, cost))

 

# # testing

# print('\nTesting ------------')
# cost = model.evaluate(X_test, Y_test, batch_size=40)
# print('test cost:', cost)
# W, b = model.layers[0].get_weights()
# print('Weights=', W, '\nbiases=', b)

 

# # predict

# Y_pred = model.predict(X_test)
# plt.scatter(X_test, Y_test)
# plt.plot(X_test, Y_pred,'r')
# plt.show()


model.fit(X_train,Y_train,epochs=10)
#model.predict(X_test)

scores = model.evaluate(X_test, Y_test,verbose=1,batch_size=100)
print(scores)



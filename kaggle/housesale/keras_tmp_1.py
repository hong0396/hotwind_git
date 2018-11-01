import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Activation
from keras.layers.core import Dense, Dropout,Activation
from sklearn import preprocessing
import numpy as np
from sklearn.cross_validation import train_test_split


train=pd.read_csv('train.csv', index_col=0)
test=pd.read_csv('test.csv', index_col=0)


TARGET = 'SalePrice'

FEATURES = train.columns.drop([TARGET])
y = np.log(train[TARGET].values)
train=train.drop('SalePrice', axis=1)
all_date=pd.concat([train,test], axis=0)



f_cat = all_date[FEATURES].select_dtypes(include=['object']).columns
f_num = all_date[FEATURES].select_dtypes(exclude=['object']).columns


# Replace NAs
all_date[f_num] = all_date[f_num].fillna(all_date[f_num].mean())
all_date[f_cat] = all_date[f_cat].fillna('?')


dummy_cat = pd.get_dummies(all_date[f_cat])
all_date = pd.DataFrame(preprocessing.scale(all_date[f_num]), columns=f_num)
dummy_cat.index=all_date.index
#all_date = all_date.join(dummy_cat)
all_date = pd.concat([all_date, dummy_cat], axis=1, ignore_index=True)
X = all_date.values
X_train, X_test, y_train, y_test = train_test_split(X[0:train.shape[0]], y, test_size=0.1, random_state=0)

model = Sequential()

model.add(Dense(200, init='normal',activation='softmax',input_dim=X.shape[1]))
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


model.fit(X_train,y_train,epochs=10)
#model.predict(X_test)

scores = model.evaluate(X_test, y_test,verbose=1,batch_size=100)
print(scores)


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



def create_nn_model(input_dim, activation, layers, optimizer):
    model = Sequential()
    l_num = 0
    for l in layers:
        if l_num == 0:
            model.add(Dense(l, input_dim=input_dim, activation=activation, init='he_normal'))
        else:
            model.add(Dense(l, activation=activation, init='he_normal'))
        l_num = l_num + 1
    model.compile(optimizer=optimizer, loss='mse')
    return model









#数据处理
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


#模型
nnet_params = []
nnet_score = []
activations = ['softplus', 'relu', 'linear']
optimizers = ['adagrad', 'rmsprop', 'adam']
layers = [[150, 1], [150, 50, 1], [200, 100, 50, 1]]



for a in activations:
    for o in optimizers:
        for l in layers:
            model = create_nn_model(input_dim=X.shape[1], activation=a, layers=l, optimizer=o)
            fit = model.fit(X_train, y_train, batch_size=100, nb_epoch=100, validation_split=0.1, verbose=0)
            score = np.sqrt(model.evaluate(X_test, y_test))
            #print("\nActivation: {} Optimizer: {} Layers: {} Score: {}\n".format(a, o, l, score))
            nnet_params.append(str(a) + '-' + str(o) + '-' + str(l))
            nnet_score.append(score)
print(nnet_score)
res = pd.DataFrame({'params': nnet_params, 'score': nnet_score})
res.sort_values(['score'], ascending=True, inplace=True)
# print(res)
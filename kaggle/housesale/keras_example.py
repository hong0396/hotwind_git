# 加载数据分析常用库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import preprocessing
sns.set_style('darkgrid')


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


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')


ID = 'Id'
TARGET = 'SalePrice'

FEATURES = train.columns.drop([ID, TARGET])
y = np.log(train[TARGET].values)
train=train.drop('SalePrice',axis=1)
all_date=pd.concat([train,test],axis=0)



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


activations = ['softplus', 'relu', 'linear']
# It seems 'softsign', 'tanh','sigmoid', 'hard_sigmoid' do not perform well on the House Prices data
# activations = ['softplus', 'relu', 'linear','softsign', 'tanh','sigmoid', 'hard_sigmoid']
# optimizers = ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']
optimizers = ['adagrad', 'rmsprop', 'adam']
layers = [[150, 1], [150, 50, 1], [200, 100, 50, 1]]
X_train, X_test, y_train, y_test = train_test_split(X[0:train.shape[0]], y, test_size=0.1, random_state=0)
nnet_params = []
nnet_score = []


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
# res.to_csv('./keras_params_.csv')
# f, ax = plt.subplots(figsize=(20, 15))
# sns.set(style="whitegrid")
# ax = sns.barplot(x='score', y='params', data=res, label="Keras parameters - RMSE")
# ax.set(xlabel='RMSE', ylabel='Parameters')



model = create_nn_model(input_dim=X.shape[1], activation='linear', layers=[200, 100, 50, 1], optimizer='adam')
fit = model.fit(X_train, y_train, batch_size=100, nb_epoch=100, validation_split=0.1, verbose=0)

result=model.predict(X[train.shape[0]:])
result=result.flatten()
zz=[pow(np.e,x) for x in result]

df=pd.DataFrame({ 'Id': [x for x in range(1461,2920)],
                   'SalePrice': zz })
df.to_csv('over.csv',index=False)

# 作者：五长生
# 链接：https://www.jianshu.com/p/c84538580c2a
# 來源：简书
# 简书著作权归作者所有，任何形式的转载都请联系作者获得授权并注明出处。
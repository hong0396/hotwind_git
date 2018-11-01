import bulbea as bb
import numpy as np
import pandas as pd
from bulbea.learn.models import RNN
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as pplt
from bulbea.learn.evaluation import split

# share=pd.read_csv('aapl.csv')
share = bb.Share('YAHOO', 'GOOGL')

Xtrain, Xtest, ytrain, ytest = split(share, 'Close', normalize = True)

Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))
Xtest  = np.reshape( Xtest, ( Xtest.shape[0],  Xtest.shape[1], 1))

rnn = RNN([1, 100, 100, 1])
rnn.fit(Xtrain, ytrain)


p = rnn.predict(Xtest)
mean_squared_error(ytest, p)


pplt.plot(ytest)
pplt.plot(p)
pplt.show()



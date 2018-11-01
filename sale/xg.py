from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sqlalchemy import create_engine
import db_table 
from sklearn.linear_model import RidgeCV


sql='''select  * from 
dbo.hw_da_model_solor_data_20180510_only_input_TEMP_v2_0000
'''
df=db_table.con(sql)

dg=df
y=df['sal_amt']
X=dg.drop(['sal_amt'], axis = 1)
#corrmat = X.corr()

X = X.fillna(X.mean())
#X.isnull().sum().sum

X=X.values
y=y.values


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


test_scores=[]
params = [1,2,3,4,5]
for param in params :
    clf = XGBRegressor(max_depth=param)
    #test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_score = cross_val_score(clf, X_train, y_train, cv=10)
    test_scores.append(test_score)

for score in test_scores:
	print(score)



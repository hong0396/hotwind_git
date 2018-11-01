import pandas as pd
import numpy as np 
import datetime
import time
import xarray as xr
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
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sqlalchemy import create_engine
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from  sklearn.metrics  import  average_precision_score 
from sklearn.metrics import r2_score  
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import ElasticNet

def nptopd(x, y):
    test=x.tolist()
    predict=y.tolist()
    li=[]
    li.append(test)
    li.append(predict)
    #print(li)
    #lo=[i for i in map(list, zip(*li))]
    df=pd.DataFrame(li)
    df=df.T
    #df.dropna(axis = 1)
    return df





all=pd.read_csv('train_all.csv')
all=all[all!=0]
all=all.dropna(thresh=10)
all=all.dropna()
all=all.fillna(0)

all['li_123_tmp']=all['li_123_tmp']*3
all=all
y=all['li_123_tmp'].values
X=all.drop(['li_123_tmp'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


ElasticNetCV = ElasticNetCV(cv=15)
ElasticNetCV.fit(X, y)
print(ElasticNetCV.alpha_)
alpha=ElasticNetCV.alpha_


clf = ElasticNet(alpha=alpha, l1_ratio=0.6, max_iter=8000)  
clf.fit(X_train, y_train)
print(clf.coef_)  # 系数
print(clf.intercept_)  # 常量
li=list(clf.coef_)
y_pred_enet=clf.predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)
print('R2='+str(r2_score_enet))













ridgecv = LassoCV(alphas=[900,10000], max_iter=8000)
#ridgecv = LassoCV(alphas=[0.008, 0.009, 0.01, 0.011, 0.012])
ridgecv.fit(X, y)
print(ridgecv.alpha_)
alpha=ridgecv.alpha_
clf = linear_model.Lasso(alpha=alpha, max_iter=8000)  
clf.fit(X_train, y_train)
clf_predict = clf.predict(X_test)

r2=r2_score(y_test, clf_predict)
df=nptopd(y_test, clf_predict)
# print(clf.coef_)  # 系数
# print(clf.intercept_)  # 常量
print(r2)




#ridgecv = RidgeCV(alphas=[0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 20, 100])
ridgecv = RidgeCV(alphas=[0.008, 0.009, 0.01, 0.011, 0.012])
ridgecv.fit(X_train, y_train)
print(ridgecv.alpha_)
alpha=ridgecv.alpha_

clf = linear_model.Ridge(alpha=alpha)  
#第一种的
clf.fit(X_train, y_train)
clf_predict = clf.predict(X_test)
RMSE = np.sqrt(mean_squared_error(y_test,clf_predict))
print(RMSE)
MSE = mean_squared_error(y_test, clf_predict)
print(MSE)


#accuracy_score(y_test,clf_predict) #只能分类
#average_precision_score(y_test,clf_predict)




#loss  = make_scorer(my_custom_loss_func, greater_is_better=False)
#score = make_scorer(my_custom_loss_func, greater_is_better=True)
#print(loss(clf,X_test, y_test))
#print(score(clf,X_test, y_test))



#第二种
 
test_scores=cross_val_score(clf, X, y, cv=10, scoring='r2') 
#test_scores=cross_val_score(clf, X, y, cv=5, scoring='mean_absolute_error')
print(test_scores)
test_scores=cross_val_score(clf, X, y, cv=10) 
#scores = cross_val_score(clf, X, y, cv=10, scoring='neg_mean_squared_error') 
print(test_scores)



test_scores=[]
params = [1]
for param in params :
    clf = XGBRegressor(max_depth=2)
    #test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_score = cross_val_score(clf, X, y, cv=10)
    test_scores.append(test_score)

for score in test_scores:
	print(score)
	per=sum(score)/len(score)
	print(per)








# all.to_csv('tmp.csv')
# print(all)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sqlalchemy import create_engine
from sklearn.metrics import fbeta_score, make_scorer
import db_table 
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from  sklearn.metrics  import  average_precision_score 
from sklearn.metrics import r2_score  


def my_custom_loss_func(ground_truth, predictions):
    diff = np.abs(ground_truth - predictions).max()
    return np.log(1 + diff)



sql='''select  * from 
dbo.hw_da_model_solor_data_20180510_only_input_TEMP_v2_0000
'''
df=db_table.con(sql)
#print(df)
#df.to_csv('a.csv')

#print(df.columns.values.tolist())

dg=df
y=df['sal_amt']
X=dg.drop(['sal_amt'], axis = 1)
#corrmat = X.corr()

X = X.fillna(X.mean())
#X.isnull().sum().sum

X=X.values
y=y.values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)


ridgecv = LassoCV(alphas=[ 20, 30, 35,  40, 45, 50, 100], max_iter=5000)
#ridgecv = LassoCV(alphas=[0.008, 0.009, 0.01, 0.011, 0.012])
ridgecv.fit(X, y)
print(ridgecv.alpha_)
alpha=ridgecv.alpha_

clf = linear_model.Lasso(alpha=alpha, max_iter=5000)  


#第一种的
clf.fit(X_train, y_train)
clf_predict = clf.predict(X_test)
#RMSE = np.sqrt(mean_squared_error(y_test,clf_predict))
#print(RMSE)
r2=r2_score(y_test, clf_predict)

#numpy转为dataframe
test=y_test.tolist()
predict=clf_predict.tolist()
li=[]
li.append(test)
li.append(predict)
df=pd.DataFrame(li)
df=df.T
print(df)
df.to_csv('b.csv')
print(r2)

#MSE = mean_squared_error(y_test, clf_predict)
#print(MSE)


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





#all_df = all_df.drop(missing[missing>1].index,1)
#all_dummy_df = pd.get_dummies(all_df)


#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

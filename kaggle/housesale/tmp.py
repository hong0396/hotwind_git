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

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')




#X_test=test.drop(['SalePrice'], axis = 1)

#X_test= pd.get_dummies(X_test)
#X.isnull().sum().sum()
#X_test = X.fillna(X_test.mean()).values



y=train['SalePrice']
X=train.drop(['SalePrice'], axis = 1)


#print(X['MSSubClass'].dtypes)
X['MSSubClass'] = X['MSSubClass'].astype(str)
#pd.get_dummies(X['MSSubClass'], prefix='MSSubClass')
X= pd.get_dummies(X)
print(X.isnull().sum().sum())
X = X.fillna(X.mean())
print(X.isnull().sum().sum())


# numeric_cols = X.columns[X.dtypes != 'object']
# numeric_col_means = X.loc[:, numeric_cols].mean()
# numeric_col_std = X.loc[:, numeric_cols].std()
# X.loc[:, numeric_cols] = (X.loc[:, numeric_cols] - numeric_col_means)/numeric_col_std



X=X.values

y = np.log1p(y)
y = y.values
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)




# ridgecv = LassoCV(alphas=[ 20, 30, 35,  40, 45, 50, 100, 110, 120], max_iter=5000)
# #ridgecv = LassoCV(alphas=[0.008, 0.009, 0.01, 0.011, 0.012])
# ridgecv.fit(X, y)
# print(ridgecv.alpha_)
# alpha=ridgecv.alpha_

# clf = linear_model.Lasso(alpha=alpha, max_iter=5000)  



# clf.fit(X_train, y_train)
# clf_predict = clf.predict(X_test)
# #RMSE = np.sqrt(mean_squared_error(y_test,clf_predict))
# #print(RMSE)

# r2=r2_score(y_test, clf_predict)
# print(r2)

#test_score=cross_val_score(clf, X, y, cv=10, scoring='r2') 
#print(test_score)



ridgecv = RidgeCV(alphas=[0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 20, 100])
ridgecv.fit(X_train, y_train)
print(ridgecv.alpha_)
alpha=ridgecv.alpha_

clf = linear_model.Ridge(alpha=alpha)  



# y_predict = np.expm1(clf.predict(X_test))

# lasso_preds = np.expm1(model_lasso.predict(X_test))
# xgb_preds = np.expm1(model_xgb.predict(X_test))
# predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})
# predictions.plot(x = "xgb", y = "lasso", kind = "scatter")
# plt.show()






rmse= np.sqrt(-cross_val_score(clf, X, y, scoring="neg_mean_squared_error", cv = 10))
print(rmse)

# var = 'GrLivArea'
# data = pd.concat([train['SalePrice'], train[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
# #plt.show()



# test_scores=[]
# params = [1]
# for param in params :
#     clf = XGBRegressor(max_depth=2)
#     #test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
#     test_score = cross_val_score(clf, X, y, cv=10)
#     test_scores.append(test_score)

# for score in test_scores:
# 	print(score)
# 	per=sum(score)/len(score)
# 	print(per)



# ridgecv = RidgeCV(alphas=[0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 20, 100])
# ridgecv.fit(X_train, y_train)
# print(ridgecv.alpha_)
# alpha=ridgecv.alpha_

# clf = linear_model.Ridge(alpha=alpha)  



# #test_scores=cross_val_score(clf, X, y, cv=10, scoring='r2') 
# #test_scores=cross_val_score(clf, X, y, cv=5, scoring='mean_absolute_error')
# #test_scores=cross_val_score(clf, X, y, cv=10) 
# #scores = cross_val_score(clf, X, y, cv=10, scoring='neg_mean_squared_error') 
# print(test_scores)
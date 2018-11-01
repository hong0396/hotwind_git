import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import linear_model
import nptodafram
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
import csv
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor
def my_custom_loss_func(ground_truth, predictions):
    diff = np.abs(ground_truth - predictions).max()
    return np.log(1 + diff)


#lis=["sal_amt","sal_prm_amt","sal_qty","type_99_flag","type_0_flag","type_500_flag","type_123_flag","type_total888_flag","the_new_counts","the_old_counts","the_avg_price","the_new_discount","the_old_distcout","temperature_max","temperature_min","bad_weather_flag","good_weather_flag","the_new_pro_color_id_counts","the_old_pro_color_id_counts","the_new_the_pro_discount_avg","the_old_the_pro_discount_avg","sal_amt_prop_piecearea_nm_0","org_id_counts_piecearea_nm_diff_0","org_id_counts_piecearea_nm_0","sal_amt_piecearea_nm_0","sal_amt_norm999","sal_amt_last_year_norm999","sal_amt_last_month_01_norm999","sal_amt_last_year_month_02_norm999","sal_amt_piecearea_nm_last_year_0_norm999","sal_amt_piecearea_nm_last_month_01_norm999","sal_amt_piecearea_nm_last_year_month_02_norm99","type_lastyear_123_flag_norm999","type_123_flag_norm999","the_new_the_sal_amt_last_year_0_norm999","the_old_the_sal_amt_last_year_0_norm999","F_sal_amt_last_year_0_norm999","X_SHOE_sal_amt_last_year_0_norm999","female_sal_amt_last_year_0_norm999","male_sal_amt_last_year_0_norm999","the_new_the_sal_amt_last_month_1_norm999","the_old_the_sal_amt_last_month_1_norm999","F_sal_amt_last_month_1_norm999","X_SHOE_sal_amt_last_month_1_norm999","female_sal_amt_last_month_1_norm999","male_sal_amt_last_month_1_norm999"]
# lis=["sal_amt"]

f = open('D:/工作/预测第二阶段/train.csv')
h = open('D:/工作/预测第二阶段/test.csv')
train=pd.read_csv(f)
test=pd.read_csv(h)


y=train['sal_amt'].values
X=train.drop(['sal_amt'], axis = 1)
print(X.shape[0])
num=X.shape[0]
all_data = pd.concat([X, test])


# print(all_data.index)
# print(X.index)
columns=all_data.columns.values.tolist()
all_data = all_data.fillna(all_data.mean()).values



X_train, X_test, y_train, y_test = train_test_split(all_data[:num], y, test_size = 0.2)


ElasticNetCV = ElasticNetCV(cv=15, alphas=[0.01, 0.09, 0.1, 0.2, 0.3, 0.5, 1])
ElasticNetCV.fit(all_data[:num], y)
print(ElasticNetCV.alpha_)
alpha=ElasticNetCV.alpha_

clf = ElasticNet(alpha=alpha, l1_ratio=0.7, max_iter=8000)  
clf.fit(X_train, y_train)
print(clf.coef_)  # 系数
print(clf.intercept_)  # 常量
li=list(clf.coef_)
y_pred_enet=clf.predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)
print(r2_score_enet)
####################################

ridgecv = RidgeCV(alphas=[0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 20, 100])
ridgecv.fit(X_train, y_train)
print(ridgecv.alpha_)
alpha=ridgecv.alpha_

clf = linear_model.Ridge(alpha=alpha, max_iter=8000)  
clf.fit(X_train, y_train)
print(clf.coef_)  # 系数
print(clf.intercept_)  # 常量
li=list(clf.coef_)
y_pred_enet=clf.predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)
print(r2_score_enet)











# ridgecv = LassoCV(alphas=[20, 30, 35,  40, 45, 50, 100, 110, 120, 130, 140], max_iter=8000)
# #ridgecv = LassoCV(alphas=[0.008, 0.009, 0.01, 0.011, 0.012])
# ridgecv.fit(all_data[:num], y)
# # print(ridgecv.alpha_)
# alpha=ridgecv.alpha_
# clf = linear_model.Lasso(alpha=alpha, max_iter=8000)  
# # print(clf.coef_)  # 系数
# # print(clf.intercept_)  # 常量
# # li=list(clf.coef_)


# test_scores=cross_val_score(clf,all_data[:num], y, cv=10, scoring='r2') 
# #test_scores=cross_val_score(clf, X, y, cv=5, scoring='mean_absolute_error')
# print(test_scores.mean())
# # clf_predict = clf.predict(   )



















# r2=r2_score(y_test, clf_predict)
# df=nptodafram.nptopd(y_test, clf_predict)
# print(clf.coef_)  # 系数
# print(clf.intercept_)  # 常量
# li=list(clf.coef_)





# dataframe = pd.DataFrame({'columns':columns,'shu':li})
# dataframe.to_csv("test.csv",index=False,sep=',')
# # li = list(map(lambda x:[x],li))  
# # with open("test.csv","w") as csvfile: 
# #     writer = csv.writer(csvfile)
# #     for i in li:
# #         writer.writerow(i)


# #df.to_csv('mo.csv')
# #print(df)
# print(r2)





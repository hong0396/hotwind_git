import pandas as pd
import numpy as np
import seaborn as sns
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
import matplotlib.pyplot as plt  
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor
from sklearn import preprocessing 
def my_custom_loss_func(ground_truth, predictions):
    diff = np.abs(ground_truth - predictions).max()
    return np.log(1 + diff)


#lis=["sal_amt","sal_prm_amt","sal_qty","type_99_flag","type_0_flag","type_500_flag","type_123_flag","type_total888_flag","the_new_counts","the_old_counts","the_avg_price","the_new_discount","the_old_distcout","temperature_max","temperature_min","bad_weather_flag","good_weather_flag","the_new_pro_color_id_counts","the_old_pro_color_id_counts","the_new_the_pro_discount_avg","the_old_the_pro_discount_avg","sal_amt_prop_piecearea_nm_0","org_id_counts_piecearea_nm_diff_0","org_id_counts_piecearea_nm_0","sal_amt_piecearea_nm_0","sal_amt_norm999","sal_amt_last_year_norm999","sal_amt_last_month_01_norm999","sal_amt_last_year_month_02_norm999","sal_amt_piecearea_nm_last_year_0_norm999","sal_amt_piecearea_nm_last_month_01_norm999","sal_amt_piecearea_nm_last_year_month_02_norm99","type_lastyear_123_flag_norm999","type_123_flag_norm999","the_new_the_sal_amt_last_year_0_norm999","the_old_the_sal_amt_last_year_0_norm999","F_sal_amt_last_year_0_norm999","X_SHOE_sal_amt_last_year_0_norm999","female_sal_amt_last_year_0_norm999","male_sal_amt_last_year_0_norm999","the_new_the_sal_amt_last_month_1_norm999","the_old_the_sal_amt_last_month_1_norm999","F_sal_amt_last_month_1_norm999","X_SHOE_sal_amt_last_month_1_norm999","female_sal_amt_last_month_1_norm999","male_sal_amt_last_month_1_norm999"]
# lis=["sal_amt"]


f = open('E:\\tempppppppppp_dierban.csv')
# f = open('E:\\tempppppppppp_bubaohanwuliao.csv')
# h = open('D:/工作/预测第二阶段/9月/test.csv')

train=pd.read_csv(f)
ass=train.copy()
print(ass.columns)
# test=pd.read_csv(h)
print('---------第1步------------')

# train = train.fillna(train.mean()) #add this
# # test = test.fillna(test.mean()) #add this
# print('---------第2步------------')
train=train.drop(['date','last_date','discount','price','styleid_pre','sizeid_pre'], axis = 1)
print('---------第2步------------')


data_train=train.copy()
df=data_train.copy()
# dummies_2 = pd.get_dummies(data_train['attrib_id2'], prefix= 'attrib_id2')
# dummies_3 = pd.get_dummies(data_train['attrib_id3'], prefix= 'attrib_id3')
# dummies_4 = pd.get_dummies(data_train['attrib_id4'], prefix= 'attrib_id4')
# dummies_5 = pd.get_dummies(data_train['attrib_id5'], prefix= 'attrib_id5')
# dummies_7 = pd.get_dummies(data_train['attrib_id7'], prefix= 'attrib_id7')
# dummies_6 = pd.get_dummies(data_train['attrib_id6'], prefix= 'attrib_id6')
# dummies_Sex = pd.get_dummies(data_train['sizeid'], prefix= 'sizeid')
# dummies_col = pd.get_dummies(data_train['colorid'], prefix= 'colorid')
# df = pd.concat([data_train, dummies_col, dummies_Sex,dummies_2,dummies_3 ,dummies_4,dummies_5,dummies_6, dummies_7 ], axis=1)
# df.drop(['attrib_id2','attrib_id3','colorid','attrib_id4','attrib_id5',  'sizeid', 'attrib_id7','attrib_id6'], axis=1, inplace=True)
print('---------第3步------------')

df = df.dropna(axis=0, how='all')
x_df=df.drop('qry', axis=1)
y=df['qry']


x=preprocessing.scale(x_df)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)



# ElasticNetCV = ElasticNetCV(cv=15, alphas=[ 0.03, 0.02,0.01,  0.2, 0.3], max_iter=8000)
ElasticNetCV = ElasticNetCV(cv=15, alphas=np.linspace(0.001, 1),max_iter=8000)
ElasticNetCV.fit(x, y)
print(ElasticNetCV.alpha_)
alpha=ElasticNetCV.alpha_


print('---------第4步------------')
clf = ElasticNet(alpha=alpha, l1_ratio=0.7, max_iter=8000)  
clf.fit(X_train, y_train)
print(clf.coef_)  # 系数
print(clf.intercept_)  # 常量
li=list(clf.coef_)
col=x_df.columns.tolist()
d = pd.DataFrame({'col':col,'li':li})
d.to_csv('xi.csv')
y_pred_enet=clf.predict(x)
r2_score_enet = r2_score(y, y_pred_enet)
yuce = pd.DataFrame({'y':y,'y_pred_enet':y_pred_enet})
yuce['org_id']=ass['org_id']
yuce['styleid']=ass['styleid_pre']
yuce['colorid']=ass['colorid']
yuce['sizeid'] =ass['sizeid_pre']
yuce.to_csv("yuce.csv",index=False)
print(r2_score_enet)


print('---------第5步------------')






# df.to_csv('E:\\temptemp.csv')
# print(df)








# sns.set()
# sns.distplot(df_train['qry'], fit=norm);
# # fig = plt.figure()
# plt.show()
# res = stats.probplot(df_train['qry'], plot=plt)


# preprocessing.scale(train[])


# sns.set()
# cols = ['qry', 'saledays', 'attrib_id4','attrib_id6']
# sns.pairplot(train[cols], size = 2.5)
# plt.show();






# print(abs(train.corr()).sort_values("qry",ascending=False)["qry"])

# y=train['sal_amt'].values
# X=train.drop(['sal_amt'], axis = 1)
# print(X.shape[0])
# num=X.shape[0]
# all_data = pd.concat([X, test])


# # print(all_data.index)
# # print(X.index)
# columns=all_data.columns.values.tolist()
# all_data=all_data.drop(['last_month'], axis = 1)
# all_data=all_data.drop(['last_year'], axis = 1)
# all_data=all_data.drop(['last_year_last_month'], axis = 1)
# # all_data=all_data.drop(['last_month'], axis = 1)
# col=all_data.columns.tolist()
# # print(all_data.columns)
# # sns.set()
# # sns.pairplot(all_data)

# sa=all_data[num:]
# st=all_data[:num]
# all_data = all_data.fillna(all_data.mean()).values


# X_train, X_test, y_train, y_test = train_test_split(all_data[:num], y, test_size = 0.2)



# # ElasticNetCV = ElasticNetCV(cv=15, alphas=[0.08, 0.09, 0.1, 0.11, 0.12, 0.01, 0.09,  0.2, 0.3, 0.4, 0.5, 0.6, 0.65,0.55, 0.57, 0.575, 0.58,  1])
# ElasticNetCV = ElasticNetCV(cv=15, alphas=np.linspace(0.5, 0.7))
# ElasticNetCV.fit(all_data[:num], y)
# print(ElasticNetCV.alpha_)
# alpha=ElasticNetCV.alpha_


# clf = ElasticNet(alpha=alpha, l1_ratio=0.85, max_iter=8000)  
# clf.fit(all_data[:num], y)
# print(clf.coef_)  # 系数
# print(clf.intercept_)  # 常量
# li=list(clf.coef_)
# d = pd.DataFrame({'col':col,'li':li})
# d.to_csv('xishu.csv')
# y_pred_enet=clf.predict(X_test)
# r2_score_enet = r2_score(y_test, y_pred_enet)
# print(r2_score_enet)
# sns.set()
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
# plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# sns.set(font='SimHei')  # 解决Seaborn中文显示问题


# dataa = pd.DataFrame({'真实销售额': y_test, '预测销售额': y_pred_enet})
# sns.lmplot(x="真实销售额", y="预测销售额", data=dataa)
# plt.xlabel("真实销售额",fontsize=12)
# plt.ylabel("预测销售额",fontsize=12)
# # plt.show()
# print(r2_score_enet)
# val_pred=clf.predict(all_data[num:])
# aa=pd.Series(val_pred)

# ss=pd.DataFrame()
# all_data[num:]
# ss['storeid']=sa['storeid']
# ss['year']=sa['year']
# ss['month']=sa['month']
# ss['pred']=aa
# ss.to_csv('a.csv')
# print(type(val_pred))

# # fig=plt.figure(figsize=(100, 100)) 

# val_pred=clf.predict(all_data[:num])
# aa=pd.Series(val_pred)
# ss=pd.DataFrame()
# ss['storeid']=st['storeid']
# ss['year']=st['year']
# ss['month']=st['month']
# ss['sal_amt']=train['sal_amt']
# ss['pred']=aa

# ss.to_csv('sum.csv')


# nu=0
# li_x=[]
# li_y=[]
# def elasticnet(valu):
#     global nu
#     global fig
#     global alpha
#     print('+++++++++++++++++++++++++++++++++++++++++')
#     print(valu)
#     clf = ElasticNet(alpha=alpha, l1_ratio=valu, max_iter=8000)  
#     clf.fit(X_train, y_train)
#     print(clf.coef_)  # 系数
#     print(clf.intercept_)  # 常量
#     f = open('test3.txt', 'a+')
#     f.writelines(str(clf.coef_)+'\n')
#     f.writelines(str(clf.intercept_)+'\n')
#     f.close()
#     li=list(clf.coef_)
#     y_pred_enet=clf.predict(X_test)
#     r2_score_enet = r2_score(y_test, y_pred_enet)
#     dataa = pd.DataFrame({'sal_amt': y_test, 'pred': y_pred_enet})
#     n=nu+1
#     # plt.subplot(8,8,n)
#     # fig=plt.figure(figsize=(100,100))
#     # fig.add_subplot(8,8,n)
#     # sns.jointplot(x="sal_amt", y="pred", data=dataa, kind='reg')
#     # plt.plot(dataa['sal_amt'], dataa['pred'])
#     # plt.title('Interesting Graph',loc ='left') 
#     # plt.annotate('local max', xy=(1, 2), xytext=(3, 1.5))
#     # plt.text(9, 1, 'test', ha='left')
#     # sns.lmplot(x="sal_amt", y="pred", data=dataa)
#     # plt.show()
#     li_x.append(valu)
#     li_y.append(r2_score_enet)
#     nu=nu+1
#     print(r2_score_enet)

# ####################################


# for i in np.linspace(0, 1):
#     elasticnet(i)


# fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)    
# plt.show()


# sns.set()
# # tips = sns.load_dataset("tips")
# # print(tips)
# sns.lmplot(x="sal_amt_last_month", y="sal_amt", kind="scatter", data=train);
# # sns.relplot()
# plt.plot(li_x, li_y)
# plt.show()



print('-----------------------------------------------------------------')
ridgecv = RidgeCV(alphas=np.linspace(0.01, 0.8))
ridgecv.fit(x, y)
print(ridgecv.alpha_)
alpha=ridgecv.alpha_

clf = linear_model.Ridge(alpha=alpha, max_iter=8000)  
clf.fit(X_train, y_train)
print(clf.coef_)  # 系数
print(clf.intercept_)  # 常量
li=list(clf.coef_)
y_pred_enet=clf.predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)
dataa = pd.DataFrame({'sal_amt': y_test, 'pred': y_pred_enet})
# sns.set()
# sns.pairplot(dataa)
# sns.residplot(x="sal_amt", y="pred", data=dataa)
# plt.show()
print(r2_score_enet)




print('-----------------------------------------------------------------')
LassoCV = LassoCV(alphas=np.linspace(0.01, 0.8))
LassoCV.fit(x, y)
print(LassoCV.alpha_)
alpha=LassoCV.alpha_


clf = linear_model.Lasso(alpha=alpha, max_iter=8000)  
clf.fit(X_train, y_train)
print(clf.coef_)  # 系数
print(clf.intercept_)  # 常量
li=list(clf.coef_)
y_pred_enet=clf.predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)
dataa = pd.DataFrame({'sal_amt': y_test, 'pred': y_pred_enet})
# sns.set()
# sns.pairplot(dataa)
# sns.residplot(x="sal_amt", y="pred", data=dataa)
# plt.show()
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



test_scores=[]
params = [1,2,3,4,5]
for param in params :
    clf = XGBRegressor(max_depth=param)
    #test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_score = cross_val_score(clf, X_train, y_train, cv=10)
    test_scores.append(test_score)

for score in test_scores:
    print(score)




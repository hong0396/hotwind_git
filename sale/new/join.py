import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.metrics import r2_score  

f = open('C:\\Users\\guihong\\Documents\\Git_hotwind\\sale\\new\\a.csv', encoding='UTF-8')
h = open('D:\\工作\\预测第二阶段\\比较\\实际.csv', encoding='UTF-8')
pred=pd.read_csv(f)
true=pd.read_csv(h)
re=pd.merge(pred, true,on='storeid')
y_test=re['sal_amt']
y_pred_enet=re['pred']
r2_score_enet = r2_score(y_test, y_pred_enet)
print(r2_score_enet)



g= open('D:\\工作\\预测第二阶段\\比较\\tmp.csv', encoding='UTF-8')
pr=pd.read_csv(g)
y_test=pr['sal_amt']
y_pred_enet=pr['pred']
r2_score_enet = r2_score(y_test, y_pred_enet)
print(r2_score_enet)
re.to_csv('bijiao.csv')
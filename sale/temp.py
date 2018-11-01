import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import linear_model




X = [[0, 0], [1, 1], [2, 2]]
y = [0, 1, 2]
clf = linear_model.Ridge(alpha=0.1)  # 设置k值
clf.fit(X, y)  # 参数拟合
print(clf.coef_)  # 系数
print(clf.intercept_)  # 常量
print(clf.predict([[3, 3]]))  # 求预测值
#print(clf.decision_function(X))  # 求预测，等同predict
print(clf.score(X, y))  # R^2，拟合优度
print(clf.get_params())  # 获取参数信息
print(clf.set_params(fit_intercept=False))  # 重新设置参数


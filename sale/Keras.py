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
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# from keras.models import load_model

# model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
# del model  # deletes the existing model

# # returns a compiled model
# # identical to the previous one
# model = load_model('my_model.h5')
 

sql='''select  * from 
dbo.hw_da_model_solor_data_20180510_only_input_TEMP_v2_0000

'''

lis=["sal_amt","sal_prm_amt","sal_qty","type_99_flag","type_0_flag","type_500_flag","type_123_flag","type_total888_flag","the_new_counts","the_old_counts","the_avg_price","the_new_discount","the_old_distcout","temperature_max","temperature_min","bad_weather_flag","good_weather_flag","the_new_pro_color_id_counts","the_old_pro_color_id_counts","the_new_the_pro_discount_avg","the_old_the_pro_discount_avg","sal_amt_prop_piecearea_nm_0","org_id_counts_piecearea_nm_diff_0","org_id_counts_piecearea_nm_0","sal_amt_piecearea_nm_0","sal_amt_norm999","sal_amt_last_year_norm999","sal_amt_last_month_01_norm999","sal_amt_last_year_month_02_norm999","sal_amt_piecearea_nm_last_year_0_norm999","sal_amt_piecearea_nm_last_month_01_norm999","sal_amt_piecearea_nm_last_year_month_02_norm99","type_lastyear_123_flag_norm999","type_123_flag_norm999","the_new_the_sal_amt_last_year_0_norm999","the_old_the_sal_amt_last_year_0_norm999","F_sal_amt_last_year_0_norm999","X_SHOE_sal_amt_last_year_0_norm999","female_sal_amt_last_year_0_norm999","male_sal_amt_last_year_0_norm999","the_new_the_sal_amt_last_month_1_norm999","the_old_the_sal_amt_last_month_1_norm999","F_sal_amt_last_month_1_norm999","X_SHOE_sal_amt_last_month_1_norm999","female_sal_amt_last_month_1_norm999","male_sal_amt_last_month_1_norm999"]



df=db_table.con(sql)
y=df['sal_amt']
X=df.drop(lis, axis = 1)
print(X.index)
columns=X.columns.values.tolist()
X = X.fillna(X.mean())


# plt.scatter(X[:200], Y[:200])
# plt.show()

#begin trainning

SPLIT=0.8*max_length
X_train, Y_train = X[:SPLIT], Y[:SPLIT]  
X_test, Y_test = X[SPLIT:], Y[SPLIT:]    


# single dense model

model = Sequential()
model.add(Dense(input_dim=1, units=1))
model.compile(loss='mse', optimizer='sgd')


print('Training -----------')
for step in range(10000):
    cost =model.train_on_batch(X_train.values, Y_train.values)
    if step % 50 == 0:
        print("After %d trainings,the cost: %f" % (step, cost))

 

# testing
print('\nTesting ------------')
cost = model.evaluate(X_test.values, Y_test.values, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

 

# predict
Y_pred = model.predict(X_test.values)
plt.scatter(X_test.values, Y_test.values)
plt.plot(X_test.values, Y_pred,'r')
plt.show()






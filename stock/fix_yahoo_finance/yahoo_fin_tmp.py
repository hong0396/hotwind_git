from yahoo_fin.stock_info import *
import pandas as pd
import numpy as np 
import datetime
import time

# msft_data = get_data('msft', start_date = '01/01/2016')
# msft_data.to_csv('temp.csv')
# print(msft_data.columns)



dic={}
nu=0
a=pd.read_csv('all.csv')
for i in a.code.tolist():
    time.sleep(0.5)
    try:
        print('---------------'+str(nu)+'--------------------')
        result = get_data(i, start_date = '01/01/2016')
        nu=nu+1
    except:
        continue 
    else: 
        dic.update({i: result})
        
p = pd.Panel(dic)

store = pd.HDFStore('all_fin_store.h5')
store['code'] = p


p.to_hdf('all_fin_store_p.h5','code', append=True)


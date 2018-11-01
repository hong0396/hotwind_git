from datetime import datetime
import py_yahoo_prices.price_fetcher as pf
import pandas as pd
import numpy as np 
import time

st_dt = datetime(2017, 6, 1)
comp_codes = ["IMM.L", "AAPL", "TSLA"]

dic={}
# get the raw prices from yahoo, auto retries on a 401 error
raw_prices = pf.multi_price_fetch(comp_codes, start_date=st_dt)
for i in comp_codes:
    if raw_prices.get(i) is not None:
        dic.update({i: raw_prices.get(i)})
p = pd.Panel(dic)
print(type(p.get('AAPL')))
store = pd.HDFStore('all_prices_store.h5')
store['code'] = p
p.to_hdf('all_prices_store_p.h5','code')
# p.get('AAPL').to_csv('tt1.csv')
# the parameters can be adjusted


# raw_prices = pf.multi_price_fetch(comp_codes, 
#                                  start_date=st_dt,
#                                  end_date=datetime(2018, 3, 1),
#                                  interval='1d')


# raw_prices.to_csv('tt2.csv')
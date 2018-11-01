import pandas as pd
import numpy as np 
import pandas_datareader.data as web
import datetime
import time
import fix_yahoo_finance as yf


a=pd.read_hdf('store_p.h5', 'p')
# print(a.get('CORI'))
print(a)
b=pd.read_hdf('store.h5', 'p')
print(b)














# yf.pdr_override()
 
# start=datetime.datetime(2016, 1, 1)
# end=datetime.datetime(2018, 1, 1)
# dic={}

# a=pd.read_csv('all.csv')
# for i in a.code.tolist()[:10]:
#     time.sleep(0.5)
#     result=web.get_data_yahoo(i,start,end)
#     dic.update({i: result})

# p = pd.Panel(dic)

# store = pd.HDFStore('store.h5')
# store['p'] = p


# data = {'Item1' : pd.DataFrame(np.random.randn(4, 3)), 
#         'Item2' : pd.DataFrame(np.random.randn(4, 2))}
# p = pd.Panel(data)
# ds=apple.index.tolist()
# y=apple['Adj Close'].tolist()
# df=pd.DataFrame({'ds':ds,'y':y})
	
# df.to_csv('aapl.csv')


# store = pd.HDFStore('store.h5')


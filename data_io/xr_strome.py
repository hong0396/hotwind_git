import pandas as pd
import numpy as np 
import pandas_datareader.data as web
import datetime
import time
import fix_yahoo_finance as yf
import xarray as xr

yf.pdr_override()
 
start=datetime.datetime(2016, 10, 1)
end=datetime.datetime(2018, 9, 1)

nu=0
a=pd.read_csv('all.csv')
# for n in range(len(a.code.tolist())):
#     if n//50 ==0:
#         nn=nn+1
def save(na,ma):
    dic={}
    for i in a.code.tolist()[na:ma]:
        print(i)
        time.sleep(1)
        try:
            print('---------------'+str(nu)+'--------------------')
            result=web.get_data_yahoo(i,start,end)
            nu=nu+1
        except:
            continue 
        else: 
            dic.update({i: result})
    ds=xr.Dataset(dic)  
    ds.to_netcdf('F:/saved_on_disk.nc',mode='a')
    return True

def save_0(na,ma):
    dic={}
    for i in a.code.tolist()[na:ma]:
        print(i)
        time.sleep(1)
        try:
            print('---------------'+str(nu)+'--------------------')
            result=web.get_data_yahoo(i,start,end)
            nu=nu+1
        except:
            continue 
        else: 
            dic.update({i: result})
    ds=xr.Dataset(dic)  
    ds.to_netcdf('F:/saved_on_disk.nc')
    return True

save_0(0,50)

# for n in range(len(a.code.tolist())//50):
#     if n == 0:
#         time.sleep(2)

#         save_0(n*50,(n+1)*50)
#     else:
#         time.sleep(5)
#         save(n*50,(n+1)*50)

    











# ds_disk =xr.open_dataset('saved_on_disk.nc')
# print(ds_disk['CORI'].to_pandas())


# with xr.open_dataset('saved_on_disk.nc') as ds:


# p = pd.Panel(dic)

# print(p)
# df_p=p.to_frame()

# df_p.to_csv('aaaaaa.csv')


# store = pd.HDFStore('all_store.h5')
# store['code'] = p


# p.to_hdf('all_store_p.h5','code', append=True)



# data = {'Item1' : pd.DataFrame(np.random.randn(4, 3)), 
#         'Item2' : pd.DataFrame(np.random.randn(4, 2))}
# p = pd.Panel(data)
# ds=apple.index.tolist()
# y=apple['Adj Close'].tolist()
# df=pd.DataFrame({'ds':ds,'y':y})
	
# df.to_csv('aapl.csv')


# store = pd.HDFStore('store.h5')


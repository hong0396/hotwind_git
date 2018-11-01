import pandas as pd
import numpy as np 
import datetime
import time
import xarray



def read_big_csv():
    reader = pd.read_csv(r'C:\Users\guihong\Documents\Git_hotwind\stock\fix_yahoo_finance\aaaaaa.csv',iterator=True,sep=',')
    loop = True
    chunksize = 100000
    chunks = []
    while loop:
        try:
            chunk = reader.get_chunk(chunksize)
            chunk = chunk.dropna(axis=1)
            chunk = chunk.drop_duplicates()
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped.")
    df = pd.concat(chunks,ignore_index=True)
    df = df.dropna(axis=1)
    df = df.drop_duplicates()
    return df

df=read_big_csv()
ds=df.to_xarray()
ds.to_netcdf('saved_on_disk.nc')
# .to_pandas()














# --------------------- 
# 作者：Anasta198110 
# 来源：CSDN 
# 原文：https://blog.csdn.net/Anasta198110/article/details/79590157 
# 版权声明：本文为博主原创文章，转载请附上博文链接！



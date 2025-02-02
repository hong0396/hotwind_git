import pandas as pd
import numpy as np 
import pandas_datareader.data as web
import datetime
import time, os
import fix_yahoo_finance as yf
import xarray as xr
from multiprocessing import Process, Pool
from multiprocessing import Process, Queue
import dask.dataframe as dd
from dask.distributed import Client
import dask.multiprocessing




ds_disk =xr.open_dataset('E:/save.nc')
# # ds_disk=ds_disk
# print(dir(ds_disk.to_dict().keys()))
# dic_disk=ds_disk.to_dict()
# li_code_tmp=[]
# print(dir(ds_disk))

code=ds_disk.to_dataframe().columns.tolist()

def cal_div(x,y):
    if y == 0:
       theta = 0
    else:
       theta = x/y
    return theta
def cal_get_data(code):
    li_code_tmp=[]
    li_0grow_tmp=[]
    li_1grow_tmp=[]
    li_2grow_tmp=[]
    li_3grow_tmp=[]
    li_4grow_tmp=[]
    li_5grow_tmp=[]




    li_1up_tmp=[]
    li_2up_tmp=[]
    li_3up_tmp=[]
    li_4up_tmp=[]
    li_5up_tmp=[]



    li_0max_tmp=[]
    li_1max_tmp=[]
    li_2max_tmp=[]
    li_3max_tmp=[]
    li_4max_tmp=[]
    li_5max_tmp=[]

    li_0min_tmp=[]
    li_1min_tmp=[]
    li_2min_tmp=[]
    li_3min_tmp=[]
    li_4min_tmp=[]
    li_5min_tmp=[]


    li_0max_close_tmp=[]
    li_1max_close_tmp=[]
    li_2max_close_tmp=[]
    li_3max_close_tmp=[]
    li_4max_close_tmp=[]
    li_5max_close_tmp=[]

    li_0min_close_tmp=[]
    li_1min_close_tmp=[]
    li_2min_close_tmp=[]
    li_3min_close_tmp=[]
    li_4min_close_tmp=[]
    li_5min_close_tmp=[]


    li_0max_min_tmp=[]
    li_1max_min_tmp=[]
    li_2max_min_tmp=[]
    li_3max_min_tmp=[]
    li_4max_min_tmp=[]
    li_5max_min_tmp=[]


    li_0max_min_near_tmp=[]
    li_1max_min_near_tmp=[]
    li_2max_min_near_tmp=[]
    li_3max_min_near_tmp=[]
    li_4max_min_near_tmp=[]


    li_0open_near_tmp=[]
    li_1open_near_tmp=[]
    li_2open_near_tmp=[]
    li_3open_near_tmp=[]
    li_4open_near_tmp=[]

    li_0close_near_tmp=[]
    li_1close_near_tmp=[]
    li_2close_near_tmp=[]
    li_3close_near_tmp=[]
    li_4close_near_tmp=[]


    li_1vol_tmp=[]
    li_2vol_tmp=[]
    li_3vol_tmp=[]
    li_4vol_tmp=[]
    li_5vol_tmp=[]

    li_grow_mean=[]
    li_grow_std=[]

    li_large_range_tmp=[]
    li_large_2345_range_tmp=[]


    li_123_tmp=[]
    for code_nm in code:
            zong=ds_disk.get(str(code_nm)).to_pandas().sort_index(ascending=False)
            zong = zong.dropna(axis = 0)  #删除行
            zong = zong.sort_values("time",ascending=False)
            if len(zong) > 100:
                for i in range(len(zong)-5):
                    if min(zong.iloc[i]['open'], zong.iloc[i+1]['open'],zong.iloc[i]['close'], zong.iloc[i+1]['close']) >= max(zong.iloc[i+2]['open'], zong.iloc[i+3]['open'],zong.iloc[i+4]['open'],zong.iloc[i+2]['close'], zong.iloc[i+3]['close'],zong.iloc[i+4]['close']):
                        if zong.iloc[i+2]['close'] == max(zong.iloc[i+2]['open'], zong.iloc[i+3]['open'],zong.iloc[i+4]['open'],zong.iloc[i+2]['close'], zong.iloc[i+3]['close'],zong.iloc[i+4]['close']) : 
                                    if abs((zong.iloc[i]['close'] - zong.iloc[i]['open'])/zong.iloc[i]['open']) < 0.03:
                                        if (zong.iloc[i+1]['close'] - zong.iloc[i+1]['open'])/zong.iloc[i+1]['open'] > 0 and abs((zong.iloc[i+1]['close'] - zong.iloc[i+1]['open'])/zong.iloc[i+1]['open']) < 0.03:
                                            if (zong.iloc[i+2]['close'] - zong.iloc[i+2]['open'])/zong.iloc[i+2]['open'] > 0 and abs((zong.iloc[i+2]['close'] - zong.iloc[i+2]['open'])/zong.iloc[i+2]['open']) < 0.03:                                       
                                                if abs((zong.iloc[i+3]['close'] - zong.iloc[i+3]['open'])/zong.iloc[i+3]['open'])< 0.03:
                                                    if abs((zong.iloc[i+4]['close'] - zong.iloc[i+4]['open'])/zong.iloc[i+4]['open']) < 0.03:
                                                        if min(zong.iloc[i+2]['open'], zong.iloc[i+3]['open'],zong.iloc[i+4]['open'],zong.iloc[i+2]['close'], zong.iloc[i+3]['close'],zong.iloc[i+4]['close'])>0:
                                                            if zong.iloc[i-5]['close']:

                                                                
                                                    
                                                                li_code_tmp.append(str(code_nm))
                                                              
                                                                grow0_tmp=(zong.iloc[i+0]['close'] - zong.iloc[i+0]['open'])/zong.iloc[i+0]['open'] 
                                                                grow1_tmp=(zong.iloc[i+1]['close'] - zong.iloc[i+1]['open'])/zong.iloc[i+1]['open'] 
                                                                grow2_tmp=(zong.iloc[i+2]['close'] - zong.iloc[i+2]['open'])/zong.iloc[i+2]['open'] 
                                                                grow3_tmp=(zong.iloc[i+3]['close'] - zong.iloc[i+3]['open'])/zong.iloc[i+3]['open'] 
                                                                grow4_tmp=(zong.iloc[i+4]['close'] - zong.iloc[i+4]['open'])/zong.iloc[i+4]['open'] 
                                                                grow5_tmp=(zong.iloc[i+5]['close'] - zong.iloc[i+5]['open'])/zong.iloc[i+5]['open'] 

                                                                
                                                                ll_tmp=[grow0_tmp,grow1_tmp,grow2_tmp,grow3_tmp,grow4_tmp,grow5_tmp] 

                                                                li_grow_std.append(np.std(ll_tmp,ddof=1))
                                                                li_grow_mean.append(np.mean(ll_tmp))

                                                                
                                                                close0=zong.iloc[i+0]['close']
                                                                close1=zong.iloc[i+1]['close']
                                                                close2=zong.iloc[i+2]['close']
                                                                close3=zong.iloc[i+3]['close']
                                                                close4=zong.iloc[i+4]['close']
                                                                close5=zong.iloc[i+5]['close']

                                                                open0=zong.iloc[i+0]['open']
                                                                open1=zong.iloc[i+1]['open']
                                                                open2=zong.iloc[i+2]['open']
                                                                open3=zong.iloc[i+3]['open']
                                                                open4=zong.iloc[i+4]['open']
                                                                open5=zong.iloc[i+5]['open']

                                                                min_value=min(open0,open1,open2,open3,open4,open5,close0,close1,close2,close3,close4,close5)
                                                                max_value=max(open0,open1,open2,open3,open4,open5,close0,close1,close2,close3,close4,close5)
                                                                
                                                                li_large_range_tmp.append((max_value-min_value)/min_value)

                                                                min_2345value=min(open2,open3,open4,open5,close2,close3,close4,close5)
                                                                max_2345value=max(open2,open3,open4,open5,close2,close3,close4,close5)
                                                                
                                                                li_large_2345_range_tmp.append((max_2345value-min_2345value)/min_2345value)



                                                                li_0grow_tmp.append((zong.iloc[i+0]['close'] - zong.iloc[i+0]['open'])/zong.iloc[i+0]['open'] )
                                                                li_1grow_tmp.append((zong.iloc[i+1]['close'] - zong.iloc[i+1]['open'])/zong.iloc[i+1]['open'] )
                                                                li_2grow_tmp.append((zong.iloc[i+2]['close'] - zong.iloc[i+2]['open'])/zong.iloc[i+2]['open'] )
                                                                li_3grow_tmp.append((zong.iloc[i+3]['close'] - zong.iloc[i+3]['open'])/zong.iloc[i+3]['open'] )
                                                                li_4grow_tmp.append((zong.iloc[i+4]['close'] - zong.iloc[i+4]['open'])/zong.iloc[i+4]['open'] )
                                                                li_5grow_tmp.append((zong.iloc[i+5]['close'] - zong.iloc[i+5]['open'])/zong.iloc[i+5]['open'] )


                                                                




                                                                li_1up_tmp.append((zong.iloc[i+1]['open'] - zong.iloc[i]['close'])/zong.iloc[i]['close']) 
                                                                li_2up_tmp.append((zong.iloc[i+2]['open'] - zong.iloc[i+1]['close'])/zong.iloc[i+1]['close']) 
                                                                li_3up_tmp.append((zong.iloc[i+3]['open'] - zong.iloc[i+2]['close'])/zong.iloc[i+2]['close']) 
                                                                li_4up_tmp.append((zong.iloc[i+4]['open'] - zong.iloc[i+3]['close'])/zong.iloc[i+3]['close'])
                                                                li_5up_tmp.append((zong.iloc[i+5]['open'] - zong.iloc[i+4]['close'])/zong.iloc[i+4]['close'])


                                                                


                                                                li_0max_tmp.append((zong.iloc[i+0]['high'] - zong.iloc[i+0]['open'])/zong.iloc[i+0]['open'] )
                                                                li_1max_tmp.append((zong.iloc[i+1]['high'] - zong.iloc[i+1]['open'])/zong.iloc[i+1]['open'] )
                                                                li_2max_tmp.append((zong.iloc[i+2]['high'] - zong.iloc[i+2]['open'])/zong.iloc[i+2]['open'] )
                                                                li_3max_tmp.append((zong.iloc[i+3]['high'] - zong.iloc[i+3]['open'])/zong.iloc[i+3]['open'] )
                                                                li_4max_tmp.append((zong.iloc[i+4]['high'] - zong.iloc[i+4]['open'])/zong.iloc[i+4]['open'] )
                                                                li_5max_tmp.append((zong.iloc[i+5]['high'] - zong.iloc[i+5]['open'])/zong.iloc[i+5]['open'] )
                                                                
                                                                li_0max_min_tmp.append((zong.iloc[i+0]['high'] - zong.iloc[i+0]['low'])/zong.iloc[i+0]['low'] )
                                                                li_1max_min_tmp.append((zong.iloc[i+1]['high'] - zong.iloc[i+1]['low'])/zong.iloc[i+1]['low'] )
                                                                li_2max_min_tmp.append((zong.iloc[i+2]['high'] - zong.iloc[i+2]['low'])/zong.iloc[i+2]['low'] )
                                                                li_3max_min_tmp.append((zong.iloc[i+3]['high'] - zong.iloc[i+3]['low'])/zong.iloc[i+3]['low'] )
                                                                li_4max_min_tmp.append((zong.iloc[i+4]['high'] - zong.iloc[i+4]['low'])/zong.iloc[i+4]['low'] )
                                                                li_5max_min_tmp.append((zong.iloc[i+5]['high'] - zong.iloc[i+5]['low'])/zong.iloc[i+5]['low'] )

                                                                li_0max_min_near_tmp.append((zong.iloc[i+1]['high'] - zong.iloc[i+0]['low'])/zong.iloc[i+0]['low'] )
                                                                li_1max_min_near_tmp.append((zong.iloc[i+2]['high'] - zong.iloc[i+1]['low'])/zong.iloc[i+1]['low'] )
                                                                li_2max_min_near_tmp.append((zong.iloc[i+3]['high'] - zong.iloc[i+2]['low'])/zong.iloc[i+2]['low'] )
                                                                li_3max_min_near_tmp.append((zong.iloc[i+4]['high'] - zong.iloc[i+3]['low'])/zong.iloc[i+3]['low'] )
                                                                li_4max_min_near_tmp.append((zong.iloc[i+5]['high'] - zong.iloc[i+4]['low'])/zong.iloc[i+4]['low'] )
                                                              

                                                                li_0open_near_tmp.append((zong.iloc[i+1]['open'] - zong.iloc[i+0]['open'])/zong.iloc[i+0]['open'] )
                                                                li_1open_near_tmp.append((zong.iloc[i+2]['open'] - zong.iloc[i+1]['open'])/zong.iloc[i+1]['open'] )
                                                                li_2open_near_tmp.append((zong.iloc[i+3]['open'] - zong.iloc[i+2]['open'])/zong.iloc[i+2]['open'] )
                                                                li_3open_near_tmp.append((zong.iloc[i+4]['open'] - zong.iloc[i+3]['open'])/zong.iloc[i+3]['open'] )
                                                                li_4open_near_tmp.append((zong.iloc[i+5]['open'] - zong.iloc[i+4]['open'])/zong.iloc[i+4]['open'] )
       
                                                                li_0close_near_tmp.append((zong.iloc[i+1]['close'] - zong.iloc[i+0]['close'])/zong.iloc[i+0]['close'] )
                                                                li_1close_near_tmp.append((zong.iloc[i+2]['close'] - zong.iloc[i+1]['close'])/zong.iloc[i+1]['close'] )
                                                                li_2close_near_tmp.append((zong.iloc[i+3]['close'] - zong.iloc[i+2]['close'])/zong.iloc[i+2]['close'] )
                                                                li_3close_near_tmp.append((zong.iloc[i+4]['close'] - zong.iloc[i+3]['close'])/zong.iloc[i+3]['close'] )
                                                                li_4close_near_tmp.append((zong.iloc[i+5]['close'] - zong.iloc[i+4]['close'])/zong.iloc[i+4]['close'] )
       





                                                                li_0min_tmp.append((zong.iloc[i+0]['low'] - zong.iloc[i+0]['open'])/zong.iloc[i+0]['open'] )
                                                                li_1min_tmp.append((zong.iloc[i+1]['low'] - zong.iloc[i+1]['open'])/zong.iloc[i+1]['open'] )
                                                                li_2min_tmp.append((zong.iloc[i+2]['low'] - zong.iloc[i+2]['open'])/zong.iloc[i+2]['open'] )
                                                                li_3min_tmp.append((zong.iloc[i+3]['low'] - zong.iloc[i+3]['open'])/zong.iloc[i+3]['open'] )
                                                                li_4min_tmp.append((zong.iloc[i+4]['low'] - zong.iloc[i+4]['open'])/zong.iloc[i+4]['open'] )
                                                                li_5min_tmp.append((zong.iloc[i+5]['low'] - zong.iloc[i+5]['open'])/zong.iloc[i+5]['open'] )

                                                                li_0max_close_tmp.append((zong.iloc[i+0]['high'] - zong.iloc[i+0]['close'])/zong.iloc[i+0]['close'] )
                                                                li_1max_close_tmp.append((zong.iloc[i+1]['high'] - zong.iloc[i+1]['close'])/zong.iloc[i+1]['close'] )
                                                                li_2max_close_tmp.append((zong.iloc[i+2]['high'] - zong.iloc[i+2]['close'])/zong.iloc[i+2]['close'] )
                                                                li_3max_close_tmp.append((zong.iloc[i+3]['high'] - zong.iloc[i+3]['close'])/zong.iloc[i+3]['close'] )
                                                                li_4max_close_tmp.append((zong.iloc[i+4]['high'] - zong.iloc[i+4]['close'])/zong.iloc[i+4]['close'] )
                                                                li_5max_close_tmp.append((zong.iloc[i+5]['high'] - zong.iloc[i+5]['close'])/zong.iloc[i+5]['close'] )
                                                                
                                                                li_0min_close_tmp.append((zong.iloc[i+0]['low'] - zong.iloc[i+0]['close'])/zong.iloc[i+0]['close'] )
                                                                li_1min_close_tmp.append((zong.iloc[i+1]['low'] - zong.iloc[i+1]['close'])/zong.iloc[i+1]['close'] )
                                                                li_2min_close_tmp.append((zong.iloc[i+2]['low'] - zong.iloc[i+2]['close'])/zong.iloc[i+2]['close'] )
                                                                li_3min_close_tmp.append((zong.iloc[i+3]['low'] - zong.iloc[i+3]['close'])/zong.iloc[i+3]['close'] )
                                                                li_4min_close_tmp.append((zong.iloc[i+4]['low'] - zong.iloc[i+4]['close'])/zong.iloc[i+4]['close'] )
                                                                li_5min_close_tmp.append((zong.iloc[i+5]['low'] - zong.iloc[i+5]['close'])/zong.iloc[i+5]['close'] )

                                                                










                                                                

                                                                # if zong.iloc[i]['Volume'] != 0 :
                                                                #     li_1vol_tmp.append((zong.iloc[i+1]['Volume'] - zong.iloc[i]['Volume'])/zong.iloc[i]['Volume']) 
                                                                # else:
                                                                #     li_1vol_tmp.append(0)
                                                                # if zong.iloc[i+1]['Volume'] != 0 :
                                                                #     li_2vol_tmp.append((zong.iloc[i+2]['Volume'] - zong.iloc[i+1]['Volume'])/zong.iloc[i+1]['Volume']) 
                                                                # else:
                                                                #     li_2vol_tmp.append(0)
                                                                # if zong.iloc[i+2]['Volume'] != 0 :
                                                                #     li_3vol_tmp.append((zong.iloc[i+3]['Volume'] - zong.iloc[i+2]['Volume'])/zong.iloc[i+2]['Volume']) 
                                                                # else:
                                                                #     li_3vol_tmp.append(0)
                                                                # if zong.iloc[i+3]['Volume'] != 0 :
                                                                #     li_4vol_tmp.append((zong.iloc[i+4]['Volume'] - zong.iloc[i+3]['Volume'])/zong.iloc[i+3]['Volume']) 
                                                                # else:
                                                                #     li_4vol_tmp.append(0)
                                                                # if zong.iloc[i+4]['Volume'] != 0 :
                                                                #     li_5vol_tmp.append((zong.iloc[i+5]['Volume'] - zong.iloc[i+4]['Volume'])/zong.iloc[i+4]['Volume']) 
                                                                # else:
                                                                #     li_5vol_tmp.append(0)
                     

                        




                                                                li_123_avg=(zong.iloc[i-1]['close']+zong.iloc[i-2]['close']+zong.iloc[i-3]['close'])/3
                                                                li_123_tmp.append((li_123_avg - min_value)/min_value)



        # del jo, zong

        # gc.collect()
 

    if len(li_code_tmp) != 0:
        tmp_df=pd.DataFrame({'code': li_code_tmp,'li_0grow_tmp': li_0grow_tmp,
            'li_1grow_tmp': li_1grow_tmp,'li_2grow_tmp': li_2grow_tmp,
            'li_3grow_tmp': li_3grow_tmp,'li_4grow_tmp': li_4grow_tmp,
            'li_5grow_tmp': li_5grow_tmp, 'li_1up_tmp': li_1up_tmp,
            'li_2up_tmp': li_2up_tmp, 'li_3up_tmp': li_3up_tmp,
            'li_4up_tmp': li_4up_tmp, 'li_5up_tmp': li_5up_tmp,
            'li_0max_tmp': li_0max_tmp,
            'li_1max_tmp': li_1max_tmp,'li_2max_tmp': li_2max_tmp,
            'li_3max_tmp': li_3max_tmp,'li_4max_tmp': li_4max_tmp,'li_5max_tmp': li_5max_tmp,
             'li_0min_tmp': li_0min_tmp,
            'li_1min_tmp': li_1min_tmp,'li_2min_tmp': li_2min_tmp,
            'li_3min_tmp': li_3min_tmp,'li_4min_tmp': li_4min_tmp,'li_5min_tmp': li_5min_tmp,

             'li_0max_min_tmp':li_0max_min_tmp,
             'li_1max_min_tmp':li_1max_min_tmp,
             'li_2max_min_tmp':li_2max_min_tmp,
             'li_3max_min_tmp':li_3max_min_tmp,
             'li_4max_min_tmp':li_4max_min_tmp,
             'li_5max_min_tmp':li_5max_min_tmp,
              
             'li_0max_min_near_tmp':li_0max_min_near_tmp,
             'li_1max_min_near_tmp':li_1max_min_near_tmp,
             'li_2max_min_near_tmp':li_2max_min_near_tmp,
             'li_3max_min_near_tmp':li_3max_min_near_tmp,
             'li_4max_min_near_tmp':li_4max_min_near_tmp,

             #    'li_0open_near_tmp':li_0open_near_tmp,
             #    'li_1open_near_tmp':li_1open_near_tmp,
             #    'li_2open_near_tmp':li_2open_near_tmp,
             #    'li_3open_near_tmp':li_3open_near_tmp,
             #    'li_4open_near_tmp':li_4open_near_tmp,

             #    'li_0close_near_tmp':li_0close_near_tmp,
             #    'li_1close_near_tmp':li_1close_near_tmp,
             #    'li_2close_near_tmp':li_2close_near_tmp,
             #    'li_3close_near_tmp':li_3close_near_tmp,
             #    'li_4close_near_tmp':li_4close_near_tmp,

            'li_large_range_tmp':li_large_range_tmp,
            'li_large_2345_range_tmp':li_large_2345_range_tmp,

            'li_0max_close_tmp': li_0max_close_tmp,
            'li_1max_close_tmp': li_1max_close_tmp,'li_2max_close_tmp': li_2max_close_tmp,
            'li_3max_close_tmp': li_3max_close_tmp,'li_4max_close_tmp': li_4max_close_tmp,'li_5max_close_tmp': li_5max_close_tmp,
             'li_0min_close_tmp': li_0min_close_tmp,
            'li_1min_close_tmp': li_1min_close_tmp,'li_2min_close_tmp': li_2min_close_tmp,
            'li_3min_close_tmp': li_3min_close_tmp,'li_4min_close_tmp': li_4min_close_tmp,'li_5min_close_tmp': li_5min_close_tmp,
            #  'li_1vol_tmp': li_1vol_tmp,
            # 'li_2vol_tmp': li_2vol_tmp, 'li_3vol_tmp': li_3vol_tmp,
            # 'li_4vol_tmp': li_4vol_tmp, 'li_5vol_tmp': li_5vol_tmp,
            'li_grow_std':li_grow_std,'li_grow_mean':li_grow_mean,
            'li_123_tmp': li_123_tmp})
    else:
        tmp_df=None  


    # order = ['code', 'li_0close_near_tmp', 'li_0grow_tmp', 'li_0max_close_tmp', 'li_0max_min_near_tmp', 'li_0max_min_tmp', 'li_0max_tmp', 'li_0min_close_tmp', 'li_0min_tmp', 'li_0open_near_tmp', 'li_123_tmp', 'li_1close_near_tmp', 'li_1grow_tmp', 'li_1max_close_tmp', 'li_1max_min_near_tmp', 'li_1max_min_tmp', 'li_1max_tmp', 'li_1min_close_tmp', 'li_1min_tmp', 'li_1open_near_tmp', 'li_1up_tmp', 'li_2close_near_tmp', 'li_2grow_tmp', 'li_2max_close_tmp', 'li_2max_min_near_tmp', 'li_2max_min_tmp', 'li_2max_tmp', 'li_2min_close_tmp', 'li_2min_tmp', 'li_2open_near_tmp', 'li_2up_tmp', 'li_3close_near_tmp', 'li_3grow_tmp', 'li_3max_close_tmp', 'li_3max_min_near_tmp', 'li_3max_min_tmp', 'li_3max_tmp', 'li_3min_close_tmp', 'li_3min_tmp', 'li_3open_near_tmp', 'li_3up_tmp', 'li_4close_near_tmp', 'li_4grow_tmp', 'li_4max_close_tmp', 'li_4max_min_near_tmp', 'li_4max_min_tmp', 'li_4max_tmp', 'li_4min_close_tmp', 'li_4min_tmp', 'li_4open_near_tmp', 'li_4up_tmp', 'li_5grow_tmp', 'li_5max_close_tmp', 'li_5max_min_tmp', 'li_5max_tmp', 'li_5min_close_tmp', 'li_5min_tmp', 'li_5up_tmp', 'li_grow_mean', 'li_grow_std']
    # tmp_df = tmp_df[order] 

    # print(tmp_df.columns.values.tolist())
    return tmp_df

def info(title):
    print(title)
    print('module name:', __name__)
    if hasattr(os, 'getppid'):  # only available on Unix
        print('parent process:', os.getppid())
    print('process id:', os.getpid())


# tmp_df=cal_get_data(code)
# tmp_df.to_csv('E:/up_analysis.csv',index=False)
from dask.distributed import Client, LocalCluster
# from multiprocessing import Process,freeze_support
if __name__ == '__main__':
    # freeze_support()
    # cluster = LocalCluster()
    # client = Client(cluster)
    # c = LocalCluster()
    # c = Client(c)
    # worker = cluster.add_worker()
    c = LocalCluster()  
    w =c.start_worker(ncores=2)  

    # w = client.start_worker(ncores=2)  

    tmp_df=cal_get_data(code)
    tmp_df.to_csv('E:/up_analysis_ce.csv',index=False)

    # client.stop_worker(w) 
    c.stop_worker(w) 





#     for key in dica:
#     if dicb.get(key):
#         dic[key]=dica[key]+dicb[key]
#     else:
#         dic[key]=dica[key]
# for key in dicb:
#     if dica.get(key):
#         pass
#     else:
#         dic[key]=dicb[key]

# --------------------- 
# 作者：galaxy_lx 
# 来源：CSDN 
# 原文：https://blog.csdn.net/galaxy_lx/article/details/80944377 
# 版权声明：本文为博主原创文章，转载请附上博文链接！
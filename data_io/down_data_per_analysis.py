<<<<<<< HEAD
import pandas as pd
=======
﻿import pandas as pd
>>>>>>> five-one
import numpy as np 
import pandas_datareader.data as web
import datetime
import time
import fix_yahoo_finance as yf
import xarray as xr


ds_disk =xr.open_dataset('E:/saved_on_disk.nc')
# # ds_disk=ds_disk
# print(dir(ds_disk.to_dict().keys()))
# dic_disk=ds_disk.to_dict()
# li_code_tmp=[]
# print(dir(ds_disk))

code=ds_disk.to_dataframe().columns.tolist()
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







li_1vol_tmp=[]
li_2vol_tmp=[]
li_3vol_tmp=[]
li_4vol_tmp=[]
li_5vol_tmp=[]

li_grow_mean=[]
li_grow_std=[]

li_123_tmp=[]
for code_nm in code:
        zong=ds_disk.get(str(code_nm)).to_pandas().sort_index(ascending=False)
        zong = zong.dropna(axis = 0)  #删除行
        # print(zong)# zong.sort_index(ascending=False) 
        if len(zong) > 100:
            for i in range(len(zong)-5):
            # if round(zong.iloc[i]['open'],2) >= round(zong.iloc[i+1]['close'],2) and round(zong.iloc[i+1]['open'],2) >= round(zong.iloc[i+2]['open'],2):
                if round(zong.iloc[i]['Open'],2) <= round(zong.iloc[i+2]['Open'],2):
                    if round(zong.iloc[i+2]['Open'],2) <= round(zong.iloc[i+3]['Open'],2):
                        if round(zong.iloc[i+3]['Open'],2) <= round(zong.iloc[i+4]['Open'],2):
                            # if round(zong.iloc[i+4]['Open'],2) <= round(zong.iloc[i+5]['Open'],2):   

                                if (zong.iloc[i]['Close'] - zong.iloc[i]['Open'])/zong.iloc[i]['Open'] <= 0:
                                    if (zong.iloc[i+1]['Close'] - zong.iloc[i+1]['Open'])/zong.iloc[i+1]['Open'] >= 0:
                                        if (zong.iloc[i+2]['Close'] - zong.iloc[i+2]['Open'])/zong.iloc[i+2]['Open'] <= 0:
                                            if (zong.iloc[i+3]['Close'] - zong.iloc[i+3]['Open'])/zong.iloc[i+3]['Open'] <= 0:
                                                if (zong.iloc[i+4]['Close'] - zong.iloc[i+4]['Open'])/zong.iloc[i+4]['Open'] <= 0:
                                                    if (zong.iloc[i+5]['Close'] - zong.iloc[i+5]['Open'])/zong.iloc[i+5]['Open'] <= 0:

                                                        if zong.iloc[i-5]['Close']:
                                                            # print(zong)
                                                            li_code_tmp.append(str(code_nm))
                                                            # li_num_tmp.append(zong.iloc[i+5]['time'])
                                                            # li_0_tmp.append(zong.iloc[i]['Close'])
                                                            grow0_tmp=(zong.iloc[i+0]['Close'] - zong.iloc[i+0]['Open'])/zong.iloc[i+0]['Open'] 
                                                            grow1_tmp=(zong.iloc[i+1]['Close'] - zong.iloc[i+1]['Open'])/zong.iloc[i+1]['Open'] 
                                                            grow2_tmp=(zong.iloc[i+2]['Close'] - zong.iloc[i+2]['Open'])/zong.iloc[i+2]['Open'] 
                                                            grow3_tmp=(zong.iloc[i+3]['Close'] - zong.iloc[i+3]['Open'])/zong.iloc[i+3]['Open'] 
                                                            grow4_tmp=(zong.iloc[i+4]['Close'] - zong.iloc[i+4]['Open'])/zong.iloc[i+4]['Open'] 
                                                            grow5_tmp=(zong.iloc[i+5]['Close'] - zong.iloc[i+5]['Open'])/zong.iloc[i+5]['Open'] 

                                                            
                                                            ll_tmp=[grow0_tmp,grow1_tmp,grow2_tmp,grow3_tmp,grow4_tmp,grow5_tmp] 

                                                            li_grow_std.append(np.std(ll_tmp, ddof=1))
                                                            li_grow_mean.append(np.mean(ll_tmp))


                                                            li_0grow_tmp.append((zong.iloc[i+0]['Close'] - zong.iloc[i+0]['Open'])/zong.iloc[i+0]['Open'] )
                                                            li_1grow_tmp.append((zong.iloc[i+1]['Close'] - zong.iloc[i+1]['Open'])/zong.iloc[i+1]['Open'] )
                                                            li_2grow_tmp.append((zong.iloc[i+2]['Close'] - zong.iloc[i+2]['Open'])/zong.iloc[i+2]['Open'] )
                                                            li_3grow_tmp.append((zong.iloc[i+3]['Close'] - zong.iloc[i+3]['Open'])/zong.iloc[i+3]['Open'] )
                                                            li_4grow_tmp.append((zong.iloc[i+4]['Close'] - zong.iloc[i+4]['Open'])/zong.iloc[i+4]['Open'] )
                                                            li_5grow_tmp.append((zong.iloc[i+5]['Close'] - zong.iloc[i+5]['Open'])/zong.iloc[i+5]['Open'] )
                                                            



                                                            li_1up_tmp.append((zong.iloc[i+1]['Open'] - zong.iloc[i]['Close'])/zong.iloc[i]['Close']) 
                                                            li_2up_tmp.append((zong.iloc[i+2]['Open'] - zong.iloc[i+1]['Close'])/zong.iloc[i+1]['Close']) 
                                                            li_3up_tmp.append((zong.iloc[i+3]['Open'] - zong.iloc[i+2]['Close'])/zong.iloc[i+2]['Close']) 
                                                            li_4up_tmp.append((zong.iloc[i+4]['Open'] - zong.iloc[i+3]['Close'])/zong.iloc[i+3]['Close'])
                                                            li_5up_tmp.append((zong.iloc[i+5]['Open'] - zong.iloc[i+4]['Close'])/zong.iloc[i+4]['Close'])
                                                            


                                                            li_0max_tmp.append((zong.iloc[i+0]['High'] - zong.iloc[i+0]['Open'])/zong.iloc[i+0]['Open'] )
                                                            li_1max_tmp.append((zong.iloc[i+1]['High'] - zong.iloc[i+1]['Open'])/zong.iloc[i+1]['Open'] )
                                                            li_2max_tmp.append((zong.iloc[i+2]['High'] - zong.iloc[i+2]['Open'])/zong.iloc[i+2]['Open'] )
                                                            li_3max_tmp.append((zong.iloc[i+3]['High'] - zong.iloc[i+3]['Open'])/zong.iloc[i+3]['Open'] )
                                                            li_4max_tmp.append((zong.iloc[i+4]['High'] - zong.iloc[i+4]['Open'])/zong.iloc[i+4]['Open'] )
                                                            li_5max_tmp.append((zong.iloc[i+5]['High'] - zong.iloc[i+5]['Open'])/zong.iloc[i+5]['Open'] )
                                                            
                                                            li_0max_min_tmp.append((zong.iloc[i+0]['High'] - zong.iloc[i+0]['Low'])/zong.iloc[i+0]['Low'] )
                                                            li_1max_min_tmp.append((zong.iloc[i+1]['High'] - zong.iloc[i+1]['Low'])/zong.iloc[i+1]['Low'] )
                                                            li_2max_min_tmp.append((zong.iloc[i+2]['High'] - zong.iloc[i+2]['Low'])/zong.iloc[i+2]['Low'] )
                                                            li_3max_min_tmp.append((zong.iloc[i+3]['High'] - zong.iloc[i+3]['Low'])/zong.iloc[i+3]['Low'] )
                                                            li_4max_min_tmp.append((zong.iloc[i+4]['High'] - zong.iloc[i+4]['Low'])/zong.iloc[i+4]['Low'] )
                                                            li_5max_min_tmp.append((zong.iloc[i+5]['High'] - zong.iloc[i+5]['Low'])/zong.iloc[i+5]['Low'] )

<<<<<<< HEAD
                                                            li_0max_min_tmp.append((zong.iloc[i+1]['High'] - zong.iloc[i+0]['Low'])/zong.iloc[i+0]['Low'] )
                                                            li_1max_min_tmp.append((zong.iloc[i+2]['High'] - zong.iloc[i+1]['Low'])/zong.iloc[i+1]['Low'] )
                                                            li_2max_min_tmp.append((zong.iloc[i+3]['High'] - zong.iloc[i+2]['Low'])/zong.iloc[i+2]['Low'] )
                                                            li_3max_min_tmp.append((zong.iloc[i+4]['High'] - zong.iloc[i+3]['Low'])/zong.iloc[i+3]['Low'] )
                                                            li_4max_min_tmp.append((zong.iloc[i+5]['High'] - zong.iloc[i+4]['Low'])/zong.iloc[i+4]['Low'] )
                                                            li_5max_min_tmp.append((zong.iloc[i+6]['High'] - zong.iloc[i+5]['Low'])/zong.iloc[i+5]['Low'] )

=======
                                                         
>>>>>>> five-one



                                                            li_0min_tmp.append((zong.iloc[i+0]['Low'] - zong.iloc[i+0]['Open'])/zong.iloc[i+0]['Open'] )
                                                            li_1min_tmp.append((zong.iloc[i+1]['Low'] - zong.iloc[i+1]['Open'])/zong.iloc[i+1]['Open'] )
                                                            li_2min_tmp.append((zong.iloc[i+2]['Low'] - zong.iloc[i+2]['Open'])/zong.iloc[i+2]['Open'] )
                                                            li_3min_tmp.append((zong.iloc[i+3]['Low'] - zong.iloc[i+3]['Open'])/zong.iloc[i+3]['Open'] )
                                                            li_4min_tmp.append((zong.iloc[i+4]['Low'] - zong.iloc[i+4]['Open'])/zong.iloc[i+4]['Open'] )
                                                            li_5min_tmp.append((zong.iloc[i+5]['Low'] - zong.iloc[i+5]['Open'])/zong.iloc[i+5]['Open'] )

                                                            li_0max_close_tmp.append((zong.iloc[i+0]['High'] - zong.iloc[i+0]['Close'])/zong.iloc[i+0]['Close'] )
                                                            li_1max_close_tmp.append((zong.iloc[i+1]['High'] - zong.iloc[i+1]['Close'])/zong.iloc[i+1]['Close'] )
                                                            li_2max_close_tmp.append((zong.iloc[i+2]['High'] - zong.iloc[i+2]['Close'])/zong.iloc[i+2]['Close'] )
                                                            li_3max_close_tmp.append((zong.iloc[i+3]['High'] - zong.iloc[i+3]['Close'])/zong.iloc[i+3]['Close'] )
                                                            li_4max_close_tmp.append((zong.iloc[i+4]['High'] - zong.iloc[i+4]['Close'])/zong.iloc[i+4]['Close'] )
                                                            li_5max_close_tmp.append((zong.iloc[i+5]['High'] - zong.iloc[i+5]['Close'])/zong.iloc[i+5]['Close'] )
                                                            
                                                            li_0min_close_tmp.append((zong.iloc[i+0]['Low'] - zong.iloc[i+0]['Close'])/zong.iloc[i+0]['Close'] )
                                                            li_1min_close_tmp.append((zong.iloc[i+1]['Low'] - zong.iloc[i+1]['Close'])/zong.iloc[i+1]['Close'] )
                                                            li_2min_close_tmp.append((zong.iloc[i+2]['Low'] - zong.iloc[i+2]['Close'])/zong.iloc[i+2]['Close'] )
                                                            li_3min_close_tmp.append((zong.iloc[i+3]['Low'] - zong.iloc[i+3]['Close'])/zong.iloc[i+3]['Close'] )
                                                            li_4min_close_tmp.append((zong.iloc[i+4]['Low'] - zong.iloc[i+4]['Close'])/zong.iloc[i+4]['Close'] )
                                                            li_5min_close_tmp.append((zong.iloc[i+5]['Low'] - zong.iloc[i+5]['Close'])/zong.iloc[i+5]['Close'] )

                                                                                                                       
                                                            

                                                            if zong.iloc[i]['Volume'] != 0 :
                                                                li_1vol_tmp.append((zong.iloc[i+1]['Volume'] - zong.iloc[i]['Volume'])/zong.iloc[i]['Volume']) 
                                                            else:
                                                                li_1vol_tmp.append(0)
                                                            if zong.iloc[i+1]['Volume'] != 0 :
                                                                li_2vol_tmp.append((zong.iloc[i+2]['Volume'] - zong.iloc[i+1]['Volume'])/zong.iloc[i+1]['Volume']) 
                                                            else:
                                                                li_2vol_tmp.append(0)
                                                            if zong.iloc[i+2]['Volume'] != 0 :
                                                                li_3vol_tmp.append((zong.iloc[i+3]['Volume'] - zong.iloc[i+2]['Volume'])/zong.iloc[i+2]['Volume']) 
                                                            else:
                                                                li_3vol_tmp.append(0)
                                                            if zong.iloc[i+3]['Volume'] != 0 :
                                                                li_4vol_tmp.append((zong.iloc[i+4]['Volume'] - zong.iloc[i+3]['Volume'])/zong.iloc[i+3]['Volume']) 
                                                            else:
                                                                li_4vol_tmp.append(0)
                                                            if zong.iloc[i+4]['Volume'] != 0 :
                                                                li_5vol_tmp.append((zong.iloc[i+5]['Volume'] - zong.iloc[i+4]['Volume'])/zong.iloc[i+4]['Volume']) 
                                                            else:
                                                                li_5vol_tmp.append(0)
                 

                    



                                                            li_123_avg=(zong.iloc[i-1]['Close']+zong.iloc[i-2]['Close']+zong.iloc[i-3]['Close'])/3
                                                            li_123_tmp.append((li_123_avg - zong.iloc[i]['Close'])/zong.iloc[i]['Close'])



    # del jo, zong

    # gc.collect()

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
     
tmp_df.to_csv('down_analysis1.csv',index=False)



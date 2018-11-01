import pandas as pd
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
li_123_tmp=[]
for code_nm in code:
        zong=ds_disk.get(str(code_nm)).to_pandas().sort_index(ascending=False)
        zong = zong.dropna(axis = 0)  #删除行
        # print(zong)# zong.sort_index(ascending=False) 
        if len(zong) > 100:
            for i in range(len(zong)-5):
                if min(zong.iloc[i]['Open'], zong.iloc[i+1]['Open'],zong.iloc[i]['Close'], zong.iloc[i+1]['Close']) >= max(zong.iloc[i+2]['Open'], zong.iloc[i+3]['Open'],zong.iloc[i+4]['Open'],zong.iloc[i+2]['Close'], zong.iloc[i+3]['Close'],zong.iloc[i+4]['Close']):
                



                # if round(zong.iloc[i]['Open'],2) >= round(zong.iloc[i+2]['Open'],2):
                #     if round(zong.iloc[i+2]['Open'],2) >= round(zong.iloc[i+3]['Open'],2):
                #         if round(zong.iloc[i+3]['Open'],2) >= round(zong.iloc[i+4]['Open'],2):
                #             # if round(zong.iloc[i+4]['Open'],2) >= round(zong.iloc[i+5]['Open'],2):      
                                

                                if abs((zong.iloc[i]['Close'] - zong.iloc[i]['Open'])/zong.iloc[i]['Open']) < 0.03:
                                    if (zong.iloc[i+1]['Close'] - zong.iloc[i+1]['Open'])/zong.iloc[i+1]['Open'] > 0 and abs((zong.iloc[i+1]['Close'] - zong.iloc[i+1]['Open'])/zong.iloc[i+1]['Open']) < 0.03:
                                        if (zong.iloc[i+2]['Close'] - zong.iloc[i+2]['Open'])/zong.iloc[i+2]['Open'] > 0 and abs((zong.iloc[i+2]['Close'] - zong.iloc[i+2]['Open'])/zong.iloc[i+2]['Open']) < 0.03:                                       
                                            if abs((zong.iloc[i+3]['Close'] - zong.iloc[i+3]['Open'])/zong.iloc[i+3]['Open'])< 0.03:
                                                if abs((zong.iloc[i+4]['Close'] - zong.iloc[i+4]['Open'])/zong.iloc[i+4]['Open']) < 0.03:
                                                   
                                                        if zong.iloc[i-5]['Close']:
                                                            li_code_tmp.append(str(code_nm))
                                                            # li_num_tmp.append(zong.iloc[i+5]['time'])
                                                            # li_0_tmp.append(zong.iloc[i]['Close'])
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
                                                            li_123_avg=(zong.iloc[i-1]['Close']+zong.iloc[i-2]['Close']+zong.iloc[i-3]['Close'])/3
                                                            li_123_tmp.append((li_123_avg - zong.iloc[i]['Close'])/zong.iloc[i]['Close'])



    # del jo, zong

    # gc.collect()

tmp_df=pd.DataFrame({'code': li_code_tmp,'li_0grow_tmp': li_0grow_tmp,
    'li_1grow_tmp': li_1grow_tmp,'li_2grow_tmp': li_2grow_tmp,
    'li_3grow_tmp': li_3grow_tmp,'li_4grow_tmp': li_4grow_tmp,
    'li_5grow_tmp': li_5grow_tmp, 'li_1up_tmp': li_1up_tmp,
    'li_2up_tmp': li_2up_tmp, 'li_3up_tmp': li_3up_tmp,
    'li_4up_tmp': li_4up_tmp, 'li_5up_tmp': li_5up_tmp,'li_123_tmp': li_123_tmp})
     
tmp_df.to_csv('Second_analysis1.csv',index=False)



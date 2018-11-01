import tushare as ts
import pandas as pd
import time

date=time.strftime('%Y-%m-%d',time.localtime(time.time()))
all_data=ts.get_today_all()

all_data=all_data[~ (all_data['name'].str.contains('ST',na=False))]

all_data=all_data.drop(['name'], axis=1)

all_data.to_csv(date+'data.csv')

grow2018_1=ts.get_growth_data(2018,1)
grow2017_2=ts.get_growth_data(2017,2)
grow2017_3=ts.get_growth_data(2017,3)
grow2017_4=ts.get_growth_data(2017,4)


grow2018_1=grow2018_1.drop(['name'], axis=1)
grow2017_2=grow2017_2.drop(['name'], axis=1)
grow2017_3=grow2017_3.drop(['name'], axis=1)
grow2017_4=grow2017_4.drop(['name'], axis=1)
# df.columns = df.columns.str.strip('$')
# df.columns = df.columns.map(lambda x:x[1:])
#a.rename(columns={'A':'a', 'B':'b', 'C':'c'}, inplace = True)
# df.rename(columns=lambda x:x.replace('$',''), inplace=True)
col=["mbrg","nprg","nav","targ","epsg","seg"]
col2017_2=map(lambda x: x+"2017_2", col) 
col2017_3=map(lambda x: x+"2017_3", col) 
col2017_4=map(lambda x: x+"2017_4", col) 
col2018_1=map(lambda x: x+"2018_1", col) 


grow2017_2.rename(columns=(dict(zip(col,col2017_2))), inplace=True) 
grow2017_3.rename(columns=(dict(zip(col,col2017_3))), inplace=True) 
grow2017_4.rename(columns=(dict(zip(col,col2017_4))), inplace=True) 
grow2018_1.rename(columns=(dict(zip(col,col2018_1))), inplace=True) 

grow1=pd.merge(grow2018_1, grow2017_2, on='code', how='outer') 
grow2=pd.merge( grow2017_3, grow2017_4,on='code', how='outer') 
grow=pd.merge( grow1, grow2,on='code', how='outer') 

s=pd.merge(grow,all_data, on='code', how='outer') 
s = s.drop_duplicates()  
s.to_csv(date+'s.csv')
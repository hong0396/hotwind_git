import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime
import fix_yahoo_finance as yf
from fbprophet import Prophet
import matplotlib.pyplot as plt
import tsfresh

yf.pdr_override()


start=datetime.datetime(2013, 1, 1)
end=datetime.datetime(2018, 9, 16)
apple=web.get_data_yahoo('AAPL',start,end)
print(apple)
ds=apple.index.tolist()
y=apple['Adj Close'].tolist()
df=pd.DataFrame({'ds':ds,'y':y})


a=tsfresh.utilities.dataframe_functions.roll_time_series(df)
print(a)
# tsfresh.utilities.dataframe_functions.make_forecasting_frame()












import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime
import time
import fix_yahoo_finance as yf
from fbprophet import Prophet
import matplotlib.pyplot as plt


yf.pdr_override()


start=datetime.datetime(1990, 1, 1)
end=datetime.datetime(2018, 9, 16)
apple=web.get_data_yahoo('GOOGL',start)
print(apple)
apple.to_csv('GOOGL.csv')
# ds=apple.index.tolist()
# y=apple['Adj Close'].tolist()
# df=pd.DataFrame({'ds':ds,'y':y})












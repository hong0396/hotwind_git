import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime
import fix_yahoo_finance as yf
from fbprophet import Prophet
import matplotlib.pyplot as plt


yf.pdr_override()


start=datetime.datetime(2018, 1, 1)
end=datetime.datetime(2018, 9, 16)
apple=web.get_data_yahoo('AAPL',start,end)
print(apple)
ds=apple.index.tolist()
y=apple['Adj Close'].tolist()
df=pd.DataFrame({'ds':ds,'y':y})




def GetProphet(df,day):
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=day)
    future.tail()
    forecast = m.predict(future)
    fig1 = m.plot(forecast)
    plt.show()
    fig2 = m.plot_components(forecast).show()
    plt.show()


GetProphet(df,30)




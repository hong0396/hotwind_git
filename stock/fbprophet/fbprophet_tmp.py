import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt


df = pd.read_csv('C:\\Users\\guihong\\Documents\\data2.csv')
print(df)



m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=365)
future.tail()
forecast = m.predict(future)
fig1 = m.plot(forecast)
plt.show()

fig2 = m.plot_components(forecast).show()
plt.show()


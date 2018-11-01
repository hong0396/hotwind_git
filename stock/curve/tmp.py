import yahoo_finance_pynterface as yahoo
import matplotlib.pyplot        as plt
import matplotlib.dates         as mdates
import matplotlib.ticker        as mticker


fig, ax = plt.subplots(1);

r = yahoo.Get.Prices("AAPL", period=['2017-09-01','2018-08-31']);
if r is not None:
    mu = r.Close.rolling(20).mean()
    sigma = r.Close.rolling(20).std()
    plt.plot(r.index.values, r.Close, color='dodgerblue', label="", zorder=30);
    plt.plot(r.index.values, mu, color='orange', label="SMA(20)", zorder=20);
    plt.fill_between(r.index.values,mu+2*sigma,mu-2*sigma, color='moccasin', label="", zorder=10);
    ax.grid(True, alpha=0.5);
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1));
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'));     
    ax.yaxis.set_major_locator(mticker.FixedLocator(range(100,300,10)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,pos: f"{x}.00 $"))
    print(r.Close);
else:
    print("something odd happened o.O");

plt.title("Apple Inc.")
plt.legend();
plt.gcf().autofmt_xdate();
plt.show();

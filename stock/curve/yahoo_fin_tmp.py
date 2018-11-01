from yahoo_fin.stock_info import *
# http://theautomatic.net/yahoo_fin-documentation/#get_analysts_info

a=get_analysts_info('nflx')
b=get_balance_sheet('nflx')
c=get_cash_flow('nflx')
d=get_quote_table('aapl')
tickers = tickers_nasdaq()
# tickers = tickers_other()
print(d)
print(tickers)
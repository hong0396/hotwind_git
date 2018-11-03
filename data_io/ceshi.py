import pandas as pd
# ,index=['c','d']
a=pd.DataFrame({'a':[1,2],'b':[3,4],'c':['c','d']})
a.set_index('c', inplace=True)
# a=a.reset_index()
a.index.rename('foo', inplace=True)
print(a)
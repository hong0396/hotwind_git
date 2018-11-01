
import pandas as pd
import matplotlib.pyplot as plt



a=pd.read_csv('C:\\Users\\guihong\\Downloads\\companylist.csv')
b=pd.read_csv('C:\\Users\\guihong\\Downloads\\companylist1.csv')
c=pd.read_csv('C:\\Users\\guihong\\Downloads\\companylist2.csv')

ll=len(a)+len(b)+len(c)
print(ll)
summ=a.append(b).append(c)
print(len(summ))
# summ=summ[summ['IPOyear']!='n/a']
summ.to_csv('C:\\Users\\guihong\\Downloads\\summ.csv')


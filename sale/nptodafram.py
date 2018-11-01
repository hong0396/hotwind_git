import pandas as pd
import numpy as np



def nptopd(x, y):
    test=x.tolist()
    predict=y.tolist()
    li=[]
    li.append(test)
    li.append(predict)
    #print(li)
    #lo=[i for i in map(list, zip(*li))]
    df=pd.DataFrame(li)
    df=df.T
    #df.dropna(axis = 1)
    return df
    
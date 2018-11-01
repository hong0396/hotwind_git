import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sqlalchemy import create_engine
import pymssql


conn=pymssql.connect(host='10.0.17.199', port='49244', database='hw_models',password='user_hwda_dbread',user='user_hwda_dbread')
cur=conn.cursor()
rows=cur.execute(sql)
for row in cur:
    print('row = %r' % (row,))
cur.close()
conn.close()

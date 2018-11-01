import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pymssql


def con(_sql):
    engine= create_engine('mssql+pymssql://user_hwda_dbread:user_hwda_dbread@10.0.17.199:49244/hw_models')
    df = pd.read_sql(sql=_sql, con=engine)
    return df
import numpy as np, pandas as pd

def ensure_datetime(df,col='date'):
 df[col]=pd.to_datetime(df[col],errors='coerce');return df

def add_julian(df,col='date'):
 df['julian_day']=df[col].dt.dayofyear;return df

#%%
import pandas as pd
import numpy as np
import yfinance as yf 
from os.path import dirname, abspath, join
import os 
import sys 



def read_csv_data(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.DATE = df.DATE.apply(pd.to_datetime) 
    return( df )

def read_folder_data(folder_path) -> list:  
    path = folder_path
    #path ='src\input variable data' # 如有需要，请修改更改相对路径
    data_path_list = [ os.path.join( path,_) for _ in  os.listdir(path)]
    return( [ read_csv_data(_) for _ in data_path_list] )



# get other asset data 
data_gold_df = yf.download(tickers="gold", interval='1d')
data_gold_df = data_gold_df[['Adj Close']]
data_gold_df.columns = ['close_gold']

data_sp500_df = yf.download(tickers="^GSPC", interval='1d')
data_sp500_df = data_sp500_df[['Adj Close']]
data_sp500_df.columns = ['close_sp500']

# get BTC data
data_BTC_df = yf.download(tickers="BTC-USD", interval='1d')
data_BTC_df['price_BTC'] = data_BTC_df['Adj Close']
data_BTC_df['range_BTC'] = data_BTC_df['High'] - data_BTC_df['Low']
data_BTC_df['return_BTC'] = np.log(data_BTC_df['Adj Close']/data_BTC_df['Adj Close'].shift(1) )
data_BTC_df['vol_BTC'] = np.power(data_BTC_df['return_BTC'] - data_BTC_df['return_BTC'].mean(),2)
data_BTC_df = data_BTC_df[['price_BTC','range_BTC', 'return_BTC', 'vol_BTC']]

#data_BTC_df.columns = ['close_BTC/USD', 'vol_BTC/USD']


# get macro data
root = dirname(dirname(dirname(__file__)))
data_macro_df_list = read_folder_data( join(root,'src\macro data') )# %%

# merge data (1)
data_df = data_BTC_df.copy()
for j in [data_gold_df, data_sp500_df]:
    data_df = pd.merge(data_df,j, left_index=True, right_index=True, how='left')

# merge data

data_df['Date'] = data_df.index.values
data_df['Y']= data_df.Date.apply(lambda x: x.year)
data_df['M']= data_df.Date.apply(lambda x: x.month)
data_df['D']= data_df.Date.apply(lambda x: x.day)
for j in data_macro_df_list:
    j['Date'] = pd.to_datetime(j.DATE)
    j['Y']= j.Date.apply(lambda x: x.year)
    j['M']= j.Date.apply(lambda x: x.month)
    if j.columns[1] not in ['CFNAI','CPALTT01USM657N','PCUOMFGOMFG','BOGZ1FA895050005Q','GDP','LREM64TTUSM156S','UMCSENT']: # 月度数据处理方式不同
        j['D']= j.Date.apply(lambda x: x.day)
        on_list = ['Y','M','D'] 
    else: 
        on_list = ['Y','M']
    j.drop(['DATE','Date'],axis=1,inplace=True)
    data_df = pd.merge(data_df, j, on=on_list, how='left').copy()
data_df.index = data_df.Date 
data_df = data_df.drop(['Y','M','D','Date'],axis=1)

#%%


# change columns name
data_df.rename({'CPALTT01USM657N':'CPI', 'PCUOMFGOMFG':'PMI', 
                'BOGZ1FA895050005Q':'Capital Cost',
                'OBMMIFHA30YF':'FHFA',
                'LREM64TTUSM156S':'employment rate',
                'T5YIFR':'inflation',
                'UMCSENT':'Consumer sentiment'
                },axis=1,inplace=True)

# change quanterly 
data_df['Capital Cost'] = data_df['Capital Cost'].fillna(method = 'ffill') 


# drop na
data_df = data_df.applymap(lambda x: float(x) if x!='.' else np.nan)

data_df['GDP'] = data_df['GDP'].fillna(method = 'ffill')
data_df = data_df.dropna()

# # add 

data_df['DGS10/DGS2'] = data_df['DGS10'] / data_df['DGS2'] 
# data_df['10Y/2Y'] = data_df['DGS10'] /data_df['DGS2']
# data_df['close_BTC/USD'] = data_df['close_BTC/USD']/1000
# data_df['vol_BTC/USD'] = data_df['vol_BTC/USD']/10e+09
# data_df['PMI'] = data_df['PMI']/10e+02
# data_df['close_sp500'] = data_df['close_sp500']/10e+03

# monthly data
data_M_df = data_df.reset_index().copy()
data_M_df = data_M_df.resample('M',on='Date').last()   
data_M_df.drop('Date', axis=1, inplace= True)

# %%

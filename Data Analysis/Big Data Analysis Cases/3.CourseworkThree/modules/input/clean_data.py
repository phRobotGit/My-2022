import pandas as pd
import numpy as np
from datetime import datetime as dt


# Clean trades 
def clean_trades(df):
    df['Price'] = df['Notional'] / df['Quantity']
    df["TimeStamp"] = df.DateTime.apply(lambda x: dt.strptime(x, "ISODate(%Y-%m-%dT%H:%M:%S.000Z)") ).values
    df['cob_date'] = df.TimeStamp.apply( lambda x: x.date() ).astype(np.datetime64)
    df.sort_index(ascending=True)
    return(df)

# Clean equity_price
def calculate_stock_daily_return(df):
    # calculate daily continous return of equity price
    df = df.sort_values(by=["cob_date"],ascending=True).reset_index()
    df['return'] = np.log(df['close'].shift()/df['close']).values
    return(df)

def clean_equity_price(df):
    df["cob_date"] = df["cob_date"].apply(lambda x: dt.strptime(str(x),"%d-%b-%Y")  ).values
    df = df.groupby(["symbol_id"],as_index=False).apply(calculate_stock_daily_return) # calculate daily continous return of equity price
    df = df.reset_index()
    df = df.drop(['level_0',"level_1","index"], axis=1)
    return(df)

# Clean portfolio_position
def clean_portfolio_position(df):
    df['cob_date'] = df['cob_date'].apply(lambda x: dt.strptime(x,"%d-%b-%Y") ).values
    return(df)
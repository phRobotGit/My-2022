import pandas as pd
import numpy as np
from datetime import datetime as dt


def find_suspect_trades(trades, equity_price):
    df = pd.merge(
            trades[['_id','Symbol','cob_date','Price','Quantity','Notional']],
            equity_price,
            left_on=['Symbol','cob_date'],
            right_on=['symbol_id','cob_date'],
            how='left')
    suspect_trades = df[ 
                        ( (df['Price'].apply(lambda x: np.floor(x)) > df['high']) |
                          (df['Price'].apply(lambda x: np.ceil(x)) < df['low']) 
                        ) 
                        & (df['cob_date'] <= np.datetime64("2021-11-12"))
                        & (df['cob_date'] >= np.datetime64("2021-11-11"))
                        ].sort_index(ascending=True)
    return(suspect_trades)


def adjust_suspect_trades(suspect_trades, trades):
    adj_trades = suspect_trades.copy() 
    adj_trades['adj_multipier'] = [100, 0.01, 0.1, 10] # adjust mutltipier
    adj_trades["Price"] = adj_trades["Price"] * adj_trades["adj_multipier"]
    adj_trades['Quantity'] = adj_trades["Quantity"] / adj_trades["adj_multipier"]
    for i in adj_trades.index:
        trades.loc[i,"Price"] = adj_trades.loc[i,"Price"]
        trades.loc[i,"Quantity"] = adj_trades.loc[i,"Quantity"]
    return(trades)
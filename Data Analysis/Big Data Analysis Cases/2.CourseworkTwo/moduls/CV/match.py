# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 19:16:07 2021

@author: Peanut Robot
"""
import pandas as pd 
import datetime as dt
import numpy as np

def matching(data_trades, data_equityPrices, id_suspect):
    x = pd.merge(data_trades[['_id','Symbol','Date','Price','Quantity']],
             data_equityPrices,
             left_on=['Symbol','Date'],
             right_on=['symbol_id','cob_date'],
             how='left')
    l = x [ ( (x['Price'].apply(lambda x: np.floor(x)) > x['high']) | (x['Price'].apply(lambda x: np.ceil(x)) < x['low'])) & 
    (x['Date'] <= np.datetime64('2021-11-12')) & 
    (x['Date'] >= np.datetime64('2021-11-11'))]['_id'].values
    
    [ id_suspect.append(i) for i in l if i not in id_suspect]
    
    return(id_suspect)
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 19:21:58 2021

@author: Peanut Robot
"""
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
import datetime as dt
import random

def detecting_outliers(data_trades, id_suspect):
    #X = data_trades[['Trader','Symbol','Quantity','Notional','TradeType','Counterparty','Price','Date']].copy()
    X_interval = data_trades[['Trader','Counterparty','Date']].copy()
    X_continuous = data_trades[['Quantity','Notional','Price']].copy().values

    
    X_interval = OneHotEncoder(handle_unknown='ignore').fit_transform(X_interval)
    X = np.c_[X_continuous,X_interval.toarray()]
    

    predict_list = []
    for count in range(30):
        random.seed( count )
        clf = IsolationForest().fit(X)
        clf.set_params(n_estimators=10)    
        predict_list.append( pd.DataFrame(clf.predict(X) ))

    prediction = pd.concat(predict_list,axis=1).applymap( lambda x: 0 if x>0 else x).mean(axis=1).values

    l = data_trades[( np.where(prediction< - 0.3, True,False).tolist() ) & 
            (data_trades['Date'] <= np.datetime64('2021-11-12')) & 
            (data_trades['Date'] >= np.datetime64('2021-11-11'))]['_id'].values
    
    [ id_suspect.append(i) for i in l if i not in id_suspect]
    
    return(id_suspect)
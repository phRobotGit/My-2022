# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 17:50:52 2021

@author: Peanut Robot
"""

import os
import configparser
import config.script_params as params
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from pymongo import MongoClient
import moduls.CV.match as module_CV 
import moduls.IF.isolation_forest as module_IF
import moduls.db.CreateTable as module_CT
import moduls.db.undate_position as module_UP
import moduls.Visualization.dash_plot as module_plot
data_file_path = r"E:/UCL BFD/big data/courework2"



if __name__ == '__main__':
    # get config 
    #config = configparser.ConfigParser()
    #config.read( os.path.join(os.getcwd(), 'config\script.config'))
    #id_suspect = config.get("config","id_suspect")
    id_suspect = []
    
    # get parameters 
    (data_equityPrices, 
     data_equityStatic,
     data_portfolioPosition,
     data_trades) = params.parse_params( data_file_path )
    
    
    # interactive visualization
    # open the url in your browser to check the plot  
    module_plot.dash_plot()
    
    
    # module CV: match datasets & cross validation || update id_suspect
    id_suspect = module_CV.matching(data_trades, data_equityPrices, id_suspect)
    
    # module IF: Isolation Forest  || update id_suspect
    id_suspect = module_IF.detecting_outliers(data_trades, id_suspect)
    
    # get data_suspect 
    data_suspect = data_trades[data_trades['_id'].apply(lambda x: x in id_suspect)].copy()
    data_suspect = data_suspect.drop(['TimeStamp','Date','Time'],axis=1)
    col_str = ['_id', 'DateTime', 'TradeId', 'Trader', 'Symbol', 'TradeType', 'Ccy', 'Counterparty']
    data_suspect[col_str]=data_suspect[col_str].astype('str').values
    
    # module db: create_sql:
    module_CT.CreateTable(data_file_path, data_suspect)

    # module db: 
    module_UP.undate(data_trades, data_portfolioPosition, data_file_path)
    
    print('finish')
    #print(data_trades)
    #print(id_suspect)
    #print(data_suspect)
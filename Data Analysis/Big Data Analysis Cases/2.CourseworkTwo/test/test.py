# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 18:33:43 2021

@author: Peanut Robot
"""
import os
import configparser
import config.script_params as params

a = params.parse_params()
config = configparser.ConfigParser()
config.read( os.path.join(os.getcwd(), 'config\script_params'))

print(a)


data_suspect = data_trades[data_trades['_id'].apply(lambda x: x in id_suspect)].copy()

                          

                          

                          

                          
data_suspect = data_suspect.drop(['TimeStamp','Date','Time'],axis=1).copy()
col_str = ['_id', 'DateTime', 'TradeId', 'Trader', 'Symbol', 'TradeType', 'Ccy', 'Counterparty']
data_suspect[col_str]=data_suspect[col_str].astype('str').values
    
    
    # module create_sql:
    module_CT.CreateTable(data_file_path, data_suspect)

    # 
    
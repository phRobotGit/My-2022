# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 17:54:38 2021

@author: Peanut Robot
"""
from pymongo import MongoClient
from sqlalchemy import create_engine
import pandas as pd 
import datetime as dt
import numpy as np

def parse_params(data_file_path):
    # get sql data
    #engine = create_engine(f"sqlite:///{data_file_path}/iftcoursework2021/000.DataBases/SQL/Equity.db")
    engine = create_engine(f"sqlite:///{data_file_path}/case 1_py/Coursework2 - case 1/moduls/db/Equity.db")
    
    #con = sqlite3.connect(r"{}/case 1_py/Coursework2 - case 1/moduls/db/Equity.db".format(directory))
    
    con = engine.connect()
    data_equityPrices = pd.read_sql_table("equity_prices", engine)  # KEY: symbol_id + cobdate  某个公司的股票信息
    data_equityPrices.cob_date = data_equityPrices.cob_date.apply(lambda x: dt.datetime.strptime(str(x),"%d-%b-%Y")  ).values
    data_equityStatic = pd.read_sql_table("equity_static", engine)  # KEY: symbol  某个公司的行业信息
    data_portfolioPosition = pd.read_sql_table("portfolio_positions", engine) # KEY: symbol + cobdate  #某个公司运营的头寸的信息
    
    # get mongo data
    con = MongoClient('mongodb://localhost')    
    data_trades = pd.DataFrame( con.db.CourseworkTwo.find({}) ) # key:  TradeID (DateTime + Symbol -- Counterparty)   # 交易信息
    data_trades['Price'] = data_trades['Notional'] / data_trades['Quantity']
    data_trades["TimeStamp"] = data_trades.DateTime.apply(lambda x: dt.datetime.strptime(x, "ISODate(%Y-%m-%dT%H:%M:%S.000Z)") ).values
    data_trades['Date'] = data_trades.TimeStamp.apply( lambda x: x.date() ).astype(np.Datetime64)
    data_trades['Time'] = data_trades.TimeStamp.apply( lambda x: x.time() )
    
    con.close()
    return( data_equityPrices, 
            data_equityStatic,
            data_portfolioPosition,
            data_trades)

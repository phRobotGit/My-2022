# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 19:49:50 2021

@author: Peanut Robot
"""
import sqlite3
import pandas as pd

def CreateTable(directory, data_suspect):
    
    con = sqlite3.connect(r"{}/case 1_py/Coursework2 - case 1/moduls/db/Equity.db".format(directory))
    
    #con = sqlite3.connect(r"{}/iftcoursework2021/000.DataBases/SQL/Equity.db".format(directory))
    sql_1 = "drop table trades_suspects"

    sql_2 = '''
     CREATE TABLE trades_suspects (
        _id TEXT PRIMARY KEY NOT NULL,
        DateTime TEXT NOT NULL,
        TradeId TEXT NOT NULL,
        Trader TEXT NOT NULL,
        Symbol TEXT NOT NULL,
        Quantity FlOAT NOT NULL,
        Notional FLOAT NOT NULL,
        TradeType TEXT NOT NULL,
        Ccy TEXT NOT NULL,
        Counterparty TEXT NOT NULL,
        Price FLOAT NOT NULL
        )
     '''

    sql_3 = '''
        INSERT INTO 
        trades_suspects( '_id', 'DateTime', 'TradeId', 'Trader', 'Symbol', 'Quantity',
                        'Notional', 'TradeType', 'Ccy', 'Counterparty', 'Price' )
        VALUES
        (?,?,?,?,?,?,?,?,?,?,?)
        '''

    try: 
        con.execute(sql_1)
    except:
        pass
    #except Exception() as e:
    #    print(e)
    try:
        cursor = con.execute(sql_2)
    except:
        pass
    #except Exception() as e:
    #    print(e)
    data_suspect.apply(lambda x: con.execute( sql_3, tuple([i for i in x]) ),axis=1 )
    con.commit()
    con.close()
    
    print('Successfully save into sqlite')
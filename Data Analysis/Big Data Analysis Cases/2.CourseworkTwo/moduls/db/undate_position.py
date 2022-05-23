# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 21:22:59 2021

@author: Peanut Robot
"""
import sqlite3
import pandas as pd 
import numpy as np

def f(df,day_s1,day_s2):
    q = float( df['net_quantity']  )
    n_q = float( df['Quantity'] )
    m = float( df['net_amount']  )
    n_m = float( df['Notional'] )
    
    return(pd.Series({
        'pos_id':df['pos_id'][:7] + day_s1 + df['pos_id'][15:],
        'cob_date': day_s2,
        'trader':df['trader'],
        'symbol':df['symbol'],
        'ccy':df['ccy'],
        'net_quantity':np.nansum( np.array( [q , n_q])),
        'net_amount':np.nansum( np.array( [m , n_m])),
    }))


def get_new_position(data_trades, data_portfolioPosition):
    df_p_10 = data_portfolioPosition[ data_portfolioPosition['cob_date'] == '10-Nov-2021'].copy()
    
    df = data_trades.groupby(['Date','Trader','Symbol','Ccy'])[['Notional','Quantity']].apply(sum).reset_index().copy()
    
    df_11 = df[(df['Date'] == np.datetime64('2021-11-11')) ].copy() 
    
    df_p_11 = pd.merge(df_p_10, 
                       df_11, 
                       left_on=['trader','symbol'], 
                       right_on=['Trader','Symbol'], 
                       how='left').apply(lambda x:f(x,'20211111','11-Nov-2021'),axis=1)
    
    df_12 = df[(df['Date'] == np.datetime64('2021-11-12')) ].copy() 

    df_p_12 = pd.merge(df_p_11, 
                       df_12, 
                       left_on=['trader','symbol'], 
                       right_on=['Trader','Symbol'], 
                       how='left').apply(lambda x:f(x,'20211112','12-Nov-2021'),axis=1)

    return( pd.concat([df_p_11, df_p_12],axis=0) )
    
def undate(data_trades,data_portfolioPosition, directory):
    df = get_new_position(data_trades, data_portfolioPosition)
    
    con = sqlite3.connect(r"{}/case 1_py/Coursework2 - case 1/moduls/db/Equity.db".format(directory))
    sql = '''
        INSERT INTO 
        portfolio_positions( 'pos_id', 'cob_date', 'trader', 'symbol', 'ccy', 'net_quantity',
       'net_amount')
        VALUES
        (?,?,?,?,?,?,?)
        '''
    #return(df)    
    df.apply(lambda x: con.execute( sql, tuple([i for i in x]) ),axis=1 )
    con.commit()
    con.close()
    print('Successfully update')
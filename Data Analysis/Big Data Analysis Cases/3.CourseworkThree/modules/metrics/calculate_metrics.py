#%%
import pandas as pd
import numpy as np

def f1(df): # metrics_by_trader
    return(pd.Series({
        'return': np.nansum(df['weight * return']),
        'net_amount': np.nansum( df['net_amount'])
    })
    )


def calculate_metrics(portfolio_position):
    portfolio_position['weight * return'] = portfolio_position["weight"] * portfolio_position['return']
    
    # by trader & date
    metrics_by_trader_sector = portfolio_position.groupby(['trader','GICSSector']).apply(f1).reset_index()
    metrics_by_trader_sector['deviation'] = metrics_by_trader_sector.groupby(['trader'])['return'].transform(np.std).values

    # by trader & sector 
    metrics_by_trader_date = portfolio_position.groupby(['cob_date','trader']).apply(f1).reset_index()
    metrics_by_trader_date['deviation'] = metrics_by_trader_date.groupby(['trader'])['return'].transform(np.std).values


    #by trader
    metrics_by_trader = metrics_by_trader_date[metrics_by_trader_date['cob_date']==np.datetime64('2021-11-12')]
    
    

    return(metrics_by_trader, metrics_by_trader_date, metrics_by_trader_sector)
    #metrics_by_trader_date = .groupby()
    #portfolio_position.groupby(['trader']).groupby(f2)
    #return_portfolio = data_portfolio_position.groupby(['cob_date','trader'])['weight * return'].apply(np.nansum)
    
    #return(return_portfolio)




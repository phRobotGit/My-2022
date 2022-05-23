#%%
import pandas as pd
import numpy as np
from datetime import datetime as dt


def f(df):
    return(pd.Series({        
        "net_quantity": np.nansum(df["Quantity"]) ,
        "net_amount": np.nansum(df["Notional"])
    }))


def update_portfolio_position(trades, portfolio_position):
    df = trades.groupby(["Ccy","cob_date","Trader","Symbol"]).apply(f)
    df = df.reset_index()
    df['pos_id'] = df.apply(
                            lambda x: "{}{}{}".format(
                                            x["Trader"],
                                            dt.strftime(x["cob_date"],"%Y%m%d"),
                                            x["Symbol"],
                                        ) ,axis=1)
    df = df[['pos_id','cob_date','Trader','Symbol','Ccy','net_quantity','net_amount']]
    df.columns = portfolio_position.columns

    portfolio_position =pd.concat([
                portfolio_position,
                df[ df['cob_date']> np.datetime64("2021-11-10") ] # update 2021-11-11 & 2021-11-12 data
            ],axis=0)   
    portfolio_position = portfolio_position[portfolio_position['cob_date'] > np.datetime64("2021-01-01")] # delete the outlier at 2020-06-20
    return(portfolio_position)



def add_columns_portfolio_position(portfolio_position, equity_price, equity_static):
    # merge portofolio_position & equity_static
    new_portfolio_position = pd.merge(
                                    portfolio_position,
                                    equity_static,
                                    left_on="symbol",
                                    right_on="symbol",
                                    how="left",
    )

    # merge portofolio_position & equity_price
    new_portfolio_position = pd.merge(
                new_portfolio_position,
                equity_price[["cob_date","symbol_id","return"]],
                left_on=["cob_date","symbol"],
                right_on=["cob_date","symbol_id"],
                how="left"
                ) 
    new_portfolio_position['sum_net_amount'] = new_portfolio_position.groupby(['cob_date','trader'])['net_amount'].transform(np.nansum)
    new_portfolio_position['weight'] = new_portfolio_position['net_amount'] / new_portfolio_position['sum_net_amount']
    
    # add_metrics
    new_portfolio_position["weight * return"] = (new_portfolio_position["weight"] * new_portfolio_position['return']).values
    new_portfolio_position['return_symbol'] = new_portfolio_position.groupby(['cob_date','trader'])['weight * return'].transform(np.nansum).values
    new_portfolio_position['deviation_symbol'] = new_portfolio_position.groupby(["trader"])['return_symbol'].transform(np.std).values

    new_portfolio_position = new_portfolio_position.drop(["symbol_id", "weight * return"],axis=1)
    new_portfolio_position = new_portfolio_position[new_portfolio_position['cob_date'] > np.datetime64("2021-01-01")] # delete the outlier at 2020-06-20
    
    return(new_portfolio_position)
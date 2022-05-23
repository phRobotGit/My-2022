#%%
from os.path import dirname, abspath, join 
import sys
from datetime import datetime as dt
from turtle import title
import pandas as pd 
from modules.tools.read_config import  read_config,read_params
from modules.tools.trim_html import trim_html
from modules.db.connect_db import connect_mongo, connect_sql
from modules.input.retrive_data import retrive_data_mongo, retrive_data_sql
from modules.input.clean_data import clean_trades,clean_portfolio_position,clean_equity_price
from modules.input.adjust_suspects import find_suspect_trades, adjust_suspect_trades
from modules.input.update_portfolio import update_portfolio_position, add_columns_portfolio_position
from modules.metrics.calculate_metrics import calculate_metrics
from modules.plot.plotly_plot import plot
from modules.html.create_html import create_home_html
# Read config & params 

def main():
    PATH_SQLDB, PATH_MONGODB = read_config()
    trader, date =read_params()

    # Connect Databases
    con_mongo = connect_mongo(PATH_MONGODB)
    con_sqlite = connect_sql(PATH_SQLDB)

    # Get Full Dataset from Databases 
    trades = retrive_data_mongo(con_mongo)
    portfolio_position, equity_price, equity_static = retrive_data_sql(con_sqlite)

    # Clean data 
    trades = clean_trades(trades)
    equity_price = clean_equity_price(equity_price)
    portfolio_position = clean_portfolio_position(portfolio_position)

    # Adjust suspects trades 
    suspect_trades = find_suspect_trades(trades, equity_price) # find suspect trades 
    trades = adjust_suspect_trades(suspect_trades, trades) # adjust suspect data in the trades dataset 

    # Update portfolio_position ( merger data & add new columns)
    portfolio_position = update_portfolio_position(trades, portfolio_position)
    portfolio_position = add_columns_portfolio_position(portfolio_position,  equity_price, equity_static)

    # calculate metrics 
    metrics_by_trader, metrics_by_trader_date, metrics_by_trader_sector = calculate_metrics(portfolio_position)


    # plot 
    table_metris, bubble, line, pie, bar_deviation, bar_return  = plot(metrics_by_trader, metrics_by_trader_date, metrics_by_trader_sector, trader, date )


    # prepare html components form home html
    table_metris_html = trim_html(table_metris.to_html())
    bubble_html = trim_html(bubble.to_html())
    line_html = trim_html(line.to_html())
    pie_html = trim_html(pie.to_html())
    bar_deviation_html = trim_html(bar_deviation.to_html())
    bar_return_html = trim_html(bar_return.to_html())

    # create home html
    create_home_html(table_metris_html, bubble_html, line_html, pie_html, bar_deviation_html, bar_return_html,  trader, date)

if __name__ =="__main__":
    main()
    print("finish")

# %%

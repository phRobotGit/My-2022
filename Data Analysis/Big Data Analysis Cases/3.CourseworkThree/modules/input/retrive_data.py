from pymongo import MongoClient
import pandas as pd
import sqlite3  


def retrive_data_mongo(con_mongo):
#  Get full NoSQL dataset.
    trades = pd.DataFrame( con_mongo.db.CourseworkThree.find({}) )
    return(trades)

def retrive_data_sql(con_sqlite):
# Get full SQL dataset.
    portfolio_position = pd.read_sql("Select * from portfolio_positions",con=con_sqlite)
    equity_price = pd.read_sql("Select * from equity_prices",con=con_sqlite)
    equity_static = pd.read_sql("Select * from equity_static",con=con_sqlite)
    return(portfolio_position, equity_price, equity_static)
# %%
from pymongo import MongoClient
import pandas as pd
import sqlite3  

def connect_mongo(path_mongodb):
# Connnect MongoDB 
    con_mongo = MongoClient(path_mongodb)
    return(con_mongo)


def connect_sql(path_sqldb):
# Connet SQLite 
    con_sqlite = sqlite3.connect(path_sqldb)
    return(con_sqlite)
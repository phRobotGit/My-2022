#%%
from os.path import dirname,join  
import configparser
import numpy as np

#%%
def read_config():
    conf = configparser.ConfigParser()
    file_path = join(dirname(dirname(dirname(__file__))),'config\script.config')
    conf.read(file_path)
    return( 
        eval(conf['config']['PATH_SQLDB']),
        eval(conf['config']['PATH_MONGODB']),
    )

def read_params():
    conf = configparser.ConfigParser()
    file_path = join(dirname(dirname(dirname(__file__))),'config\script.params')
    conf.read(file_path)

    return(
        conf['params']['trader'],
        np.datetime64( conf['params']['date'] )
    )

PATH_SQLDB, PATH_MONGODB = read_config()
trader, date =read_params()

# %%


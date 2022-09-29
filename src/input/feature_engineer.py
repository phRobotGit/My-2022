from calendar import different_locale
from logging.handlers import DEFAULT_TCP_LOGGING_PORT
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler




# 为每个时间序列提取特征
def _calculate_ts_feature(arr):
    arr = pd.Series(arr)

    ret_arr = np.log(arr / arr.shift(1) )
    ret_arr[ ret_arr == np.inf] = np.NaN # 把inf抹去
    ret_arr[ ret_arr == -np.inf] = np.NaN # 把-inf抹去
    
    return(pd.Series({
        # "mean": np.nanmean( arr ), # level (mean)
        "std": np.nanstd( arr),   # amptitude 
        "kurtosis": arr.kurtosis(), 
        "mean_change":np.nanmean( ret_arr ), # a kind of polarity 
        "std_change":np.nanmean(ret_arr), # a kind of magnitude of the changes 
        # shape (considering the dependency on the timestamp)
        
        # "polarity": 
    }))

def _extract_ts_feature(df):
    df = df.sort_index(ascending=True) # 排序
    obs_matricx = df.T.apply(lambda x: _calculate_ts_feature(x), axis=1)
    # print(df.index.get_level_values(0)[0])
    obs_row = obs_matricx.unstack(level=0)
    # print(obs_row.index.tolist)
    return( pd.Series( obs_row ) )



def extract_feature(df):
    df_all =df.copy()
    
    df_input_ts = df_all.groupby(by = df_all.index.get_level_values(0) ).apply(_extract_ts_feature)

    df_input_origin = df_all.unstack(level=1)
    #df_input_origin = df_input_origin.fillna(df_input_origin.mean()) # fill average 有没有问题？
    df_input = pd.merge(df_input_ts, df_input_origin, how='left', left_index=True, right_index=True)

    return(df_input)


def transform_feature(df):
    df_all = df.copy()
    df_all['semi-variance (down)'] = df_all['semi-variance (down)'].apply(np.log)
    
    df_all['VaR (95%)'] = df_all['VaR (95%)'].apply(lambda x: np.sign(x) * np.power( np.abs(x), 1/3))
    
    # df_all['D(Overall,E)'] = df_all['D(Overall,E)'].apply(lambda x: np.power(x, 1/3))
    # df_all['D(Overall,S)'] = df_all['D(Overall,S)'].apply(lambda x: np.power(x, 1/3))
    # df_all['D(Overall,G)'] = df_all['D(Overall,G)'].apply(lambda x: np.power(x, 1/3))
    df_all['D(ESG, VaR)'] = df_all['D(ESG, VaR)'].apply(lambda x: np.power(x, 1/3))
    return(df_all)


def transform_feature_after_extraction(df):
    df_input = df.copy()

    # 强行处理 missing value 回头有空了再来填补这里的空白
    df_input = df_input.fillna(df_input.mean(axis=0)) 
    df_input = df_input.dropna(axis=0)


    # Standardize Transformation
    df_input = pd.DataFrame(
                StandardScaler().fit_transform(df_input),
                columns= df_input.columns,
                index=df_input.index)
                
    return(df_input)



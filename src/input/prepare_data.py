
# ESG data
import pandas as pd 
import numpy as np

def get_ESG_data(file_path, START_YEAR, END_YEAR):
    
    '''
    Read ESG data + Clean ESG data (transform format and delete error value)
    '''

    df_ESG_Eikon = pd.read_csv(file_path)
    df_ESG_Eikon['Date'] = df_ESG_Eikon['Date'].apply(lambda x: pd.Timestamp(x))
    df_ESG_Eikon['Year'] = df_ESG_Eikon['Date'].apply(lambda x: x.year)
    
    # filter Year
    df_ESG_Eikon = df_ESG_Eikon[df_ESG_Eikon['Year'] >= START_YEAR]
    df_ESG_Eikon = df_ESG_Eikon[df_ESG_Eikon['Year'] <= END_YEAR]
    # df_ESG_Eikon = df_ESG_Eikon.dropna()

    df_ESG_Eikon = df_ESG_Eikon.sort_values(by=['Instrument','Year']).drop_duplicates(subset=['Instrument','Year'], keep='last')
    df_ESG_Eikon.index = pd.MultiIndex.from_frame(pd.DataFrame({
        "Symbol":df_ESG_Eikon['Instrument'].values,
        "Year":df_ESG_Eikon['Year'].values
    }))
    df_ESG_Eikon = df_ESG_Eikon.drop(['Instrument', 'Date', 'Year'], axis=1)
    df_ESG_Eikon = df_ESG_Eikon[[
                        'ESG Score', 
                        'Environmental Pillar Score',
                        'Social Pillar Score', 
                        'Governance Pillar Score',
                        ]]
    # df_ESG_Eikon = df_ESG_Eikon.dropna(subset=['ESG Score', 'Social Pillar Score', 'Governance Pillar Score'])
    return(df_ESG_Eikon)





def get_market_data(file_path):
    
    '''
    Read Market data + Clean Market data (transform format and delete error value)
    '''

    df_market_Eikon = pd.read_csv(file_path)
    df_market_Eikon = df_market_Eikon.drop_duplicates(keep='last')
    
    return(df_market_Eikon)





def semi_var(arr, mean, if_up = True):
    
    '''
    calculate the semi varinace of a series
    will be used in the function: calculate_risk_metric_features
    '''
    
    if if_up == True:
        bias_arr = (arr - mean)[(arr - mean)>0]
    elif if_up == False:
        bias_arr = (arr - mean)[(arr - mean)<0]
    return( np.mean(bias_arr**2) ) 



def calculate_risk_metric_features(df):
    
    '''
    calculate the risk metric features 
    '''

    df = df.sort_values('Month')
    df['Return'] = np.log( df['CLOSE']/df['CLOSE'].shift(1)).values
    mean = df['Return'].mean(skipna=True)
    return(pd.Series({
        'mean-return': df['Return'].mean(skipna=True),
        # 'std': df['Return'].std(), # too much noisy
        'semi-variance (down)': semi_var(df['Return'], mean, if_up=False),
        # 'semi-variance (up)': semi_var(df['Return'], mean, if_up=True), # 不需要上行风险，
        'kurtosis':df['Return'].kurtosis(),
        'skew': df['Return'].skew(),
        'VaR (95%)': np.nanquantile(df['Return'],0.05)
    }))




    


def calculate_changes(arr):
    
    '''
    calculate the changes of each feature
    will be used in the function: calculate_distance_features
    '''

    arr = pd.Series(arr)
    ret_arr = np.log(arr / arr.shift(1) )
    ret_arr[ ret_arr == np.inf] = np.NaN # 把inf抹去
    ret_arr[ ret_arr == -np.inf] = np.NaN # 把-inf抹去
    return(ret_arr)



def calculate_distance_features(df):
    '''
    calculate the distance features 
    '''
    
    df_changes = df.groupby(by = df.index.get_level_values(0)).apply(lambda df: df.apply(calculate_changes,axis=0)).copy()
    # calcualte distance 

    D_OE = ( df_changes['ESG Score'] - df_changes['Environmental Pillar Score'] )**2
    D_OS = ( df_changes['ESG Score'] - df_changes['Social Pillar Score'] )**2
    D_OG = ( df_changes['ESG Score'] - df_changes['Governance Pillar Score'] )**2


    return(pd.DataFrame({
        # "D(Overall,E)": D_OE
        # "D(Overall,S)": D_OS,
        # "D(Overall,G)": D_OG,
        # "D(Overall,ESG)": D_OE + D_OS + D_OG,
        "D(ESG, kurtosis)": ( df_changes['ESG Score'] - df_changes['kurtosis'] )**2,
        "D(ESG, VaR)": ( df_changes['ESG Score'] - df_changes['VaR (95%)'] )**2,
    }, index=df.index)
    )
from turtle import tilt
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np

def plot_top_K_ts(top_K, df_results, df_all):
    if top_K<0:
        series = df_results['Proba (is A)'].nsmallest(-1*top_K)
    if top_K>0:
        series = df_results['Proba (is A)'].nlargest(top_K)
    top_K = abs(top_K)
    symbol_list = series.index.tolist()
    proba_list = series.values.tolist()

    fig, axs = plt.subplots(2,top_K, figsize=(15,10))
    col_1 = ['Year', 'ESG Score','Environmental Pillar Score', 'Social Pillar Score','Governance Pillar Score']
    col_2 = ['Year', 'mean-return','semi-variance (down)','kurtosis', 'skew', 'VaR (95%)'] # , 'kurtosis', 'skew'
    col_3 = ['Year', 'D(ESG, VaR)'] # , 'kurtosis', 'skew'

    for i in range(top_K):
        symbol = symbol_list[i]
        
        # symbol =  df_results[df_results[('Proba (is A)','')]>0.5].index[0]
        df = df_all[df_all.index.get_level_values(0) == symbol].reset_index().copy()
        df[col_1].plot.line(x='Year', ax=axs[0,i], title = f'''{symbol}: Proba({proba_list[i]:.2%})''')
        df[col_2].plot.line(x='Year', ax=axs[1,i])
        # df[col_3].plot.line(x='Year', ax=axs[2,i])
        
    return(fig, df_all[ [ i in symbol_list for i in df_all.index.get_level_values(0) ] ] )


def plot_feature_distribution(df): 
    df = df.copy() 
    # df.drop([''])
    df[('Proba (is A)','')] = pd.cut( df['Proba (is A)'], bins=[0, 0.1, 0.2, 0.5, 1] )
    
    df_g = df.groupby([('Proba (is A)','')]).mean()
    df_g.index = ['P(Anomaly) in (0,10%]', 'P(Anomaly) in (10%,20%]', 'P(Anomaly) in (20%,50%]', 'P(Anomaly) in (50%,100%]'] 
    df_g = df_g.drop([('Anomaly_Score',''), ('Label','')],axis=1)

    fig = df_g.T.plot.bar(
            figsize=(25,10), 
            title='Figure 3: How the mean value of each feature varies between the default group and the non-default group'
    )


    df_g.columns = [f'''{a}-{b}''' for a,b in zip(df_g.columns.get_level_values(0), df_g.columns.get_level_values(1))]
    
    return(fig, df_g)
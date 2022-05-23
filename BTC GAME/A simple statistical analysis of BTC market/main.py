#%% 
from typing import Text
from pyparsing import col

from sympy import plot
from modules.input.input import data_df, data_M_df
from modules.tools.tsa_tool import plot_acf_pacf, test_autocorrelation
from modules.tools.plot_tool import *
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from vega_datasets import data
import datapane as dp
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from statsmodels.tsa import stattools
import statsmodels as sm


# prepare log data
temp = [ i for i in data_M_df.columns if i not in ['price_BTC', 'return_BTC', 'vol_BTC'] ]
data_r_M_df = data_M_df[temp].apply(lambda x: x/x.shift(1), axis=0)
data_r_M_df['vol_BTC'] = data_M_df['vol_BTC'].copy()


# ------ TSA --------
# transform data 
df = data_df.copy()
df = df.resample('2w').last()
df['return_BTC'] = df[['price_BTC']].apply(lambda x: np.log(x/x.shift(1)), axis=0) 
df['vol_BTC'] = df[['return_BTC']].apply(lambda x: np.power(x - x.mean(), 2), axis=0 )
temp_list_other = [ 'Capital Cost', 'CFNAI', 'CPI', 'GDP', 'employment rate','FHFA', 'PMI', 'Consumer sentiment']
temp_list = ['close_gold','close_sp500', 'inflation', 'DGS2', 'DGS10','DGS10/DGS2']
temp_list_r = [i+'_ret' for i in temp_list]
temp_list_v = [i+'_vol' for i in temp_list]
df[temp_list_r] = df[temp_list].apply(lambda x: np.log(x/x.shift(1)), axis=0) 
df[temp_list_v] = df[temp_list_r].apply(lambda x: np.power(x - x.mean(), 2), axis=0 )
df = df.dropna()

# test
P1 = plot_acf_pacf(df['return_BTC'], 'return_BTC') 
P2 = plot_acf_pacf(df['vol_BTC'], 'vol_BTC') 


adf_test_1 = stattools.adfuller(df['return_BTC'])  # H0 ： 存在单位根，不平稳
adf_test_2 = stattools.adfuller(df['vol_BTC'])  # H0 ： 存在单位根，不平稳


arch_test = test_autocorrelation(df['vol_BTC']) # garch

# plot agian 
P3 = plot_uni_charts(df[['return_BTC','vol_BTC','price_BTC','range_BTC']],column_num=2)


Page_TSA_stability = dp.Page(
        dp.Text("**结论**"),
        dp.Text("1.BTC收益率平稳；不对称，负面冲击更大"),
        dp.Text("2.BTC波动率平稳，无明显聚集效应；存在spike；似乎存在周期；波动率有逐渐变小的趋势"),
        dp.Text("3.BTC当日内极差较波动率存在滞后"),
        dp.Text("4.BTC价格持续上升"),
        dp.Text("启发："),
        dp.Text("1.平稳 & 周期 -> 某些规律在不断的重复 -> 可以帮助我们择时 （Q： BTC return平稳就挺奇怪的，按理来说市场应该在博弈中进化才对，不应该这么平稳）"),
        dp.Text("2.判断：价格持续上升+波动率在逐渐变小 -> 是否可以靠中性策略来取得收益？"),
        dp.Text("3.未来的工作方向： （1）验证2中判断是否成立，如果成立，寻找可以用来对冲BTC波动率的标的 （2）分解1中的周期/规律，从而进行更好的择时： 找出spike的成因，时序分解， 博弈分析"), 
        dp.Text('---'),
        dp.Text('**描述**'),
        dp.Plot(P3),
        dp.Table(df[['return_BTC','vol_BTC']].describe()),
        dp.Text('---'),
        dp.Text('''**ACF/PACF**'''),
        dp.Group(
            dp.Plot(P1),
            dp.Plot(P2),
            columns = 2 
        ),
        dp.Text(f'''**ADF test** ( H0: 存在单位根，不平稳 )'''),
        dp.Text(f'''- return{adf_test_1[-2]}'''),
        dp.Text(f'''- volatility{adf_test_2[-2]}'''),
        dp.Text(f'''BTC收益和波动率都平稳，甚至比股票等市场更平稳，说明: 1.历史在重复的发生。或者说，BTC市场中有某种不断重复的规律/周期。2.收益率和波动率都很接近白噪声，造成这个现象的原因是什么？'''),
        dp.Text(f'''---'''),
        dp.Text(f'''**ARCH text** (H0: 自相关系数都为0)'''),
        dp.Table(arch_test),
        dp.Text(f'''BTC波动率自身没有明显的聚集效应'''),
        dp.Text(f'''---'''),
        dp.Text("**Find possible pair trading**"),
        dp.Plot(plot_uni_charts(df[ ['vol_BTC']+temp_list_v],column_num=2)),
        dp.Plot(plot_uni_charts(df[ ['return_BTC']+temp_list_r],column_num=2)),
        dp.Plot(plot_uni_charts(df[ temp_list_other],column_num=2)),
        #dp.Plot(sns.pairplot(df[['vol_BTC']+temp_list_other])),
        title="TSA—平稳性分析"
    )
# -----------------------------------------

from statsmodels.tsa.api import VAR

#mdata = sm.datasets.macrodata.load_pandas().data
# make a VAR model
a_list =  ['vol_BTC']  + ['close_sp500_vol','close_gold_vol','DGS10_ret','DGS2_ret','DGS10/DGS2_ret','DGS10_vol'] + temp_list_other
#a_list =  ['vol_BTC']  + temp_list_v + temp_list_other
model = VAR(df[a_list].dropna())
results = model.fit(6)
results.summary()
results.pvalues
results.pvalues.applymap(lambda x: x<0.05 ).sum()

a_df_list=[]
a_df = pd.DataFrame(results.intercept,columns=['const']).T
a_df.columns = a_list
a_df_list.append(a_df)

for i in range(len(results.coefs)):
    b_list = [ f'''L{i+1}.{_}'''for _ in a_list]
    a_df_list.append( pd.DataFrame(results.coefs[i],columns = a_list, index=b_list) )
a_df = pd.concat(a_df_list,axis=0)
b_df = a_df.where(results.pvalues<0.10,0)
b_df = b_df.iloc[1:] # 去掉const
c_df = b_df[b_df['vol_BTC']!=0][['vol_BTC']].copy()
c_df['label'] = [ _.split('.')[0] for _ in c_df.index]
c_df['var']=[ _.split('.')[1] for _ in c_df.index]
c_df.index = pd.MultiIndex.from_frame(c_df[['label','var']])
c_df = c_df[['vol_BTC']].copy()
c_df = c_df.unstack('var').copy()
c_df['sort'] = [ int(_[1:]) for _ in c_df.index.values]
c_df.sort_values(by=['sort'],inplace=True)
c_df
#c_df.drop(['label','var'],axis=0,inplace=True)


# -- Linear -- 
from scipy import stats
import numpy as np
import statsmodels.api as sm

temp_list_other_2 =['Capital Cost',
 'CPI',
 'GDP',
 'employment rate',
 'FHFA',
 'PMI',
 'Consumer sentiment']
x = sm.add_constant(df[temp_list_other+temp_list_r], prepend=False)
y = df['return_BTC'].values
# Fit and summarize OLS model
mod = sm.GLM(y, x)
result_lm = mod.fit()
result_lm_s = result_lm.summary()
result_lm_s


Page_TSA_VAR = dp.Page(
    dp.Text("**结论**"),
    dp.Text('''1.sp500 return对BTC return有正向影响； FHFA, GDP对BTC return有负向影响'''),
    dp.Text('''2.滞后3月PMI对BTC波动率有正向影响，滞后2月CPI对BTC波动率有负向影响'''),
    dp.Text('''Q:似乎这种时序建模方法的稳定性不足，有什么更好的建模方法吗？'''),
    dp.Text('---'),
    dp.Text('**VAR 10 % significance**'),
    dp.DataTable(b_df),
    dp.Table(c_df),
    dp.Text('---'),
    dp.Text('---'),
    dp.Text('**return - macro factors**'),
    dp.Table(pd.DataFrame(result_lm_s.tables[0])),
    dp.Table(pd.DataFrame(result_lm_s.tables[1])),
    title='TSA-VAR建模'

)

#%%

# 画图 P1
a = data_df.copy()
P_EDA_1 = alt.Chart(a.reset_index()).transform_fold(
        a.columns.tolist(),
        as_ = ['name', 'value']
    ).mark_line(tooltip=alt.TooltipContent('encoding')).encode(
        x = 'Date:T',
        y = 'value:Q',
        color = 'name:N',
    ).facet(
        facet='name:N',
        columns=3
    ).resolve_scale(
        x='independent', 
        y='independent'
    ).properties(
        title='daily raw'
    )


# 画图 P2

sns.set_theme(style="white")

# Generate a large random dataset

# Compute the correlation matrix
corr = data_df.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))
# Set up the matplotlib figure
P_EDA_2 = plt.figure(figsize=(13,13))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
P_EDA_2 =sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)
P_EDA_2.set_title("X Price - range Price | volatility")

# 画图 P3 

sns.set_theme(style="white")

# Generate a large random dataset

# Compute the correlation matrix
corr = data_r_M_df.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))
# Set up the matplotlib figure
P_EDA_3 = plt.figure(figsize=(13,13))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
P_EDA_3  = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)
P_EDA_3.set_title("X growth - range growth | volatility")
P_EDA_3

# 画图 P4
a = data_r_M_df.copy()
P_EDA_4 = alt.Chart(a.reset_index()).transform_fold(
        a.columns.tolist(),
        as_ = ['name', 'value']
    ).mark_line(tooltip=alt.TooltipContent('encoding')).encode(
        x = 'Date:T',
        y = 'value:Q',
        color = 'name:N',
    ).facet(
        facet='name:N',
        columns=3
    ).resolve_scale(
        x='independent', 
        y='independent'
    ).properties(
        title='monthly log return'
    )


# 画图 P5 
temp = [i for i in data_r_M_df if i not in ['vol_BTC']]
data_v_M_df = data_r_M_df[temp].apply(lambda x: (x- x.mean())**2 ,axis=0 )
data_v_M_df['vol_BTC'] = data_r_M_df['vol_BTC']  

# Generate a large random dataset

# Compute the correlation matrix
corr = data_v_M_df.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))
# Set up the matplotlib figure
P_EDA_5 = plt.figure(figsize=(13,13))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
P_EDA_5 = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)
P_EDA_5.set_title("X volatility - volatility ")


## Report
report = dp.Report(
    dp.Page(
        dp.Text("#Volatility Spillover between Bitcoin and marco variables"),
        dp.Text("**Report-Task2 by Hang**"),
        dp.Select(blocks=[
            dp.Group(
                dp.Text("### (0) The indicators of volatility:"),
                dp.Text("- **range:** = high - low | 反应 panic "),
                dp.Text("- **volatility** = (close price - mean)^2 | 反应 volatility"),
                dp.Text("- **问题1.1**: （1）panic, 和 volatility 到底体现了怎样的博弈？"),
                dp.Text("- **问题1.2**：（2）受问题1.1启发，是否有必要对市场分层研究波动率，如：分为机构交易者行为波动率，散户行为波动率。如果要分，应该怎么分层，用什么数据？"),
                dp.Text("---"),
                dp.Text("### (1) Literature Review:"),
                dp.Text("- Dilek Teker (2020) 用时许的方法研究 石油、黄金 对加密货币波动率的影响： 发现仅 Tether和石油、黄金存在协整关系"),
                dp.Text("- E. Bouri (2018) 用时许方法发现 1.比特币回报与大多数其他资产的回报有密切相关 2.比特币的总波动溢出小于接受"),
                dp.Text( "- Hans Bystrom, D. Krygier (2018) 发现散户投资者(indicator: 谷歌搜索舆情)而非大型机构投资者是比特币波动的主要驱动力"),
                dp.Text("### (2) Infomation from the correlation:"),
                dp.Text("I"),
                dp.Text("- **极差增长率**： 1. 和黄金、股票增长率正相关；(替代效应，信心减弱) 2.和FHFA、PMI、Consumer sentimetn、employment rate、GDP增长率负相关 （基本面向好，信心增强）"),
                dp.Text("- **极差波动率**： 1. 和黄金市场波动率正相关 （替代效应，波动率互相溢出）2.和GDP、employment rate、PMI、Consumer sentiment波动率负相关 （目前无法解释，需要进一步观察，可以用半波动率来做）"),
                dp.Text("- **关于极差波动率:** 可以建立 Arma-bekk进一步观察规律"),
                dp.Text("II"),
                dp.Text("- **BTC价格波动率**: 1. 和gold, sp500增长率正相关 （替代效应），2.和DSG10,DSG2,inflation,Consumer sentiment波动率负相关 （目前无法解释，需要进一步观察，可以用半波动率来做）"),
                dp.Text("- **关于BTC价格波动率**： 可以建立arma模型"),
                dp.Text("- **问题1.3**: （1）这样做是对的吗？应该用时序的方法来建立模型吗？"),
                dp.Text("- **问题1.4**： 如果用时序做，怎么设计rolling和系统敏感度分析?"),
                dp.Text("III"),
                dp.Text("correlation 图见下，详见EDA分页"),
                dp.Plot(P_EDA_3),
                dp.Plot(P_EDA_5),


                label='Introduction'
                ),
            dp.Group(
                dp.Text("Introdc"),
                label='Data & EDA',

                ),
            dp.Group(
                dp.Text("Introdc"),
                label='Methodology'
                ),
            dp.Group(
                dp.Text("Introdc"),
                label='Experiment Result'
                ),
            dp.Group(
                dp.Text("Introdc"),
                label='Conclusion'
                ),   
            dp.Group(
                dp.Text("Reference:"),
                dp.Text('''[1] Bystrom, Hans, and Dominika Krygier. "What drives bitcoin volatility?." Available at SSRN 3223368 (2018).'''),
                dp.Text('''[2] Bouri, Elie, et al. "Spillovers between Bitcoin and other assets during bear and bull markets." Applied Economics 50.55 (2018): 5935-5949.'''),
                dp.Text('''[3] Teker, Dilek, Suat Teker, and Mustafa Ozyesil. "Macroeconomic Determinants of Cryptocurrency Volatility: Time Series Analysis." Journal of Business & Economic Policy 7.1 (2020): 65-71.'''),
                label='Reference'
                ),                 
        ], type=dp.SelectType.TABS),
        title='Overview'
    ),
    dp.Page(
        dp.Text('EDA:'),
        dp.Plot(P_EDA_1),
        dp.Text('price -> price | volatility'),
        dp.Plot(P_EDA_2),
        dp.Text('growth -> growth | volatility'),
        dp.Plot(P_EDA_3),
        dp.Plot(P_EDA_4),
        dp.Text('volatility -> volatility'),
        dp.Plot(P_EDA_5),
        title = "EDA"
    ),
    Page_TSA_stability,
    Page_TSA_VAR,
    dp.Page(
        dp.Text('Data Source:'),
        dp.DataTable(data_df),
        title = "Data"
   ),
    dp.Page(
        dp.Text("Please check the codes in the attactment"),
        title = 'Code'
    )
)
report.save("Report-task2.html")

# #%%


# import pandas as pd
# import altair as at
# at.data_transformers.disable_max_rows()

# def read_knmi_data(filename, names):
#     return pd.read_csv(filename, 
#                        comment='#',               # Skip all comments
#                        header=None,               # The text file contains no header that can be read
#                        names=names,               # Set the column names
#                        skipinitialspace=True,     # Fix the trailing spaces after the ','-separator
#                        parse_dates=[1])           # Let pandas try and transform the second column to a proper datatime object

# knmi_data = read_knmi_data('KNMI_20200218.txt', 
#                            names=['station', 'datum', 'Wsp_avg', 'Wsp_1hravg', 'Wsp_max', 
#                                                       'T_avg', 'T_min', 'T_max', 
#                                                       'Sol_duration', 'Global_radiation', 'Precip_total',
#                                                       'Precip_hrmax', 'Rel_humid', 'Evaporation'])
# knmi_data.head()
# # %%


# cor_data = (knmi_data.drop(columns=['station'])
#               .corr().stack()
#               .reset_index()     # The stacking results in an index on the correlation values, we need the index as normal columns for Altair
#               .rename(columns={0: 'correlation', 'level_0': 'variable', 'level_1': 'variable2'}))
# cor_data['correlation_label'] = cor_data['correlation'].map('{:.2f}'.format)  # Round to 2 decimal
# cor_data.head()

# cor_data = data_M_df.corr().stack().reset_index().rename(columns={0: 'correlation', 'level_0': 'variable', 'level_1': 'variable2'})
# cor_data['correlation_label'] = cor_data['correlation'].map('{:.2f}'.format)  # Round to 2 decimal
# cor_data.head()

# base = at.Chart(cor_data).encode(
#     x='variable2:O',
#     y='variable:O'    
# )

# # Text layer with correlation labels
# # Colors are for easier readability
# text = base.mark_text().encode(
#     text='correlation_label',
#     color=at.condition(
#         at.datum.correlation > 0.5, 
#         at.value('white'),
#         at.value('black')
#     )
# )

# # The correlation heatmap itself
# cor_plot = base.mark_rect().encode(
#     color='correlation:Q'
# )


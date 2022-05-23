#%%
from turtle import screensize
from matplotlib.pyplot import axis
from regex import B
from scipy.misc import electrocardiogram
from modules.tools.tools import computeIDF, computeTF, computeTFIDF, map_word_to_sent
from modules.input.input import text_list, text_name_list
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import seaborn as sns
import nltk
import matplotlib.pyplot as plt
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment import vader
from nltk.sentiment import SentimentAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from torch import negative
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import SCORERS, r2_score
import statsmodels as stats 

import statsmodels.api as sm


class LinearRegression(linear_model.LinearRegression):
    """
    LinearRegression class after sklearn's, but calculate t-statistics
    and p-values for model coefficients (betas).
    Additional attributes available after .fit()
    are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
    which is (n_features, n_coefs)
    This class sets the intercept to 0 by default, since usually we include it
    in X.
    """

    def __init__(self, *args, **kwargs):
        if not "fit_intercept" in kwargs:
            kwargs['fit_intercept'] = False
        super(LinearRegression, self)\
                .__init__(*args, **kwargs)

    def fit(self, X, y, n_jobs=1):
        self = super(LinearRegression, self).fit(X, y, n_jobs)

        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        sse = np.array([1,sse])
        se = np.array([
            np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X))))
                                                    for i in range(sse.shape[0])
                    ])
        #self.sse = sse
        #se = sse 
        self.t = self.coef_ / se
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))
        return self

# ------------------------------ Finbert 
temp_list =[]
for i in range(len(text_list)):
    inputs = tokenizer(text_list[i], padding = True, truncation = True, return_tensors='pt')

    # get output 
    outputs = model(**inputs)
    print(outputs.logits.shape) 

    # max-soft
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    print(predictions)

    # visualization 
    positive = predictions[:, 0].tolist()
    negative = predictions[:, 1].tolist()
    neutral = predictions[:, 2].tolist()

    table = {'Headline':text_name_list[i],
            "Positive":positive,
            "Negative":negative, 
            "Neutral":neutral}
        
    df = pd.DataFrame(table, columns = ["Headline", "Positive", "Negative", "Neutral"])

    temp_list.append(df)

temp_df = pd.concat(temp_list)
temp_df['Positive'] - temp_df['Negative']


#%%
# -- prepare lexicon -- 
negative_df = pd.read_csv('src/Negative.csv')
positive_df = pd.read_csv('src/Positive.csv')
lexicon_LM_neg_words_list = negative_df['LM'].tolist()
lexicon_LM_pos_words_list = positive_df['LM'].tolist()
lexicon_GI_neg_words_list = negative_df['GI'].tolist()
lexicon_GI_pos_words_list = positive_df['GI'].tolist()
lexicon_both_neg_words_list = negative_df['all'].tolist()
lexicon_both_pos_words_list = positive_df['all'].tolist()
lexicon_HW_neg_words_list = negative_df['HW'].tolist()
lexicon_HW_pos_words_list = positive_df['HW'].tolist()

# a = pd.DataFrame({
#     'N':[ i for i in lexicon_HW_neg_words_list if str(i) not in ['nan'] +lexicon_both_neg_words_list],
#     })

# b = pd.DataFrame({
#     'P':[ i for i in lexicon_HW_pos_words_list if str(i) not in ['nan'] +lexicon_both_pos_words_list],
#     })
# a.to_csv('src/N.csv')
# b.to_csv('src/P.csv')
 

# -- build indicator --
stopword_list = stopwords.words("english")

def build_sentiment_indicator(text) -> pd.DataFrame:
    # 生成Tokenization 
    tokenize_sequence = word_tokenize(text)
    # lower 
    # 移除 stop words
    tokenize_sequence = [_ for _ in tokenize_sequence if _ not in stopword_list]
    # stemming 
    tokenize_sequence = [PorterStemmer().stem(_) for _ in tokenize_sequence]
    # lemmatizating 
    tokenize_sequence = [WordNetLemmatizer().lemmatize(_) for _ in tokenize_sequence]
    # sentiment tagging | 在这一步需要更换词典
    sentiment_indicator_df = pd.DataFrame()
    a_list =[]
    b_list = []
    for i in ['LM','GI','HW','all']:
        n_negative = len([_ for _ in tokenize_sequence if _ in negative_df[i].tolist()])
        n_positive = len([_ for _ in tokenize_sequence if _ in positive_df[i].tolist()])
        if n_negative ==0:
            n_negative = 1
        if n_positive == 0 :
            n_positive =1

        # count if
        neg_list = negative_df[i].tolist()
        pos_list = positive_df[i].tolist()

        token_neg_list = [_ for _ in tokenize_sequence if _ in neg_list]
        token_pos_list = [_ for _ in tokenize_sequence if _ in pos_list]

        b = pd.Series(token_neg_list)
        b_list.append( b.value_counts() )

        a = pd.Series(token_pos_list)
        a_list.append( a.value_counts() )
        
        sentiment_indicator_df[i] = [ (n_positive - n_negative)/(n_positive + n_negative) ]
    
    b = pd.concat(b_list,axis=1)
    b.columns = ['LM','GI','HW','all']
    b.fillna(0,inplace=True)

    a = pd.concat(a_list,axis=1)
    a.columns = ['LM','GI','HW','all']
    a.fillna(0,inplace=True)
    return( a, b )

# a, b, c = build_sentiment_indicator(text_list[0])


# text = text_list[0]
# tokenize_sequence = word_tokenize(text)

#map_word_to_sent(tokenize_sequence, pos_list, neg_list)
tf_list = [ build_sentiment_indicator(_) for _ in text_list]

tf_pos_list =[]
tf_neg_list =[]
count = 0 
for (tf_pos, tf_neg) in tf_list:
    p = tf_pos.stack().reset_index() 
    p.columns = ['level_0','level_1',str(count)]
    tf_pos_list.append( p )
    n = tf_neg.stack().reset_index()
    n.columns = ['level_0','level_1',str(count)]
    tf_neg_list.append( n )
    count += 1


tf_neg_df =pd.DataFrame()
tf_pos_df = pd.DataFrame()
for i in range(len(tf_neg_list)):
    if i == 0:
        tf_neg_df = tf_neg_list[i]
        tf_pos_df = tf_pos_list[i]
    else:
        tf_neg_df = pd.merge(tf_neg_df,tf_neg_list[i], on=['level_0','level_1'],how='outer')
        tf_pos_df = pd.merge(tf_pos_df,tf_pos_list[i], on=['level_0','level_1'],how='outer')
    

col = [ i for i in tf_neg_df.columns if i not in ['level_0', 'level_1']] 
sum_neg_series = tf_neg_df[col].sum(axis=1)
sum_pos_series = tf_pos_df[col].sum(axis=1)
for i in col:
    tf_neg_df[i] = tf_neg_df[i] / sum_neg_series
    tf_pos_df[i] = tf_pos_df[i] / sum_pos_series
tf_neg_df.fillna(0, inplace=True)
tf_pos_df.fillna(0, inplace= True)

score_neg_df = tf_neg_df.groupby(['level_1']).apply(lambda x: x.sum(axis=0))
score_pos_df = tf_pos_df.groupby(['level_1']).apply(lambda x: x.sum(axis=0))

indicator_df = (score_pos_df[col] - score_neg_df[col])/(score_neg_df[col] + score_pos_df[col])
indicator_df = indicator_df.T
indicator_df.index = text_name_list
indicator_df.columns = [i +'_indicator' for i in indicator_df.columns]


indicator_df['FinBERT_indicator'] = (temp_df['Positive'] - temp_df['Negative']).values
# -------------------------------------------------------
# -------------------------------------------------------



# tf_neg_df 
# tf_neg_df.groupby(['level_0', 'level_1']).apply(lambda x: x.sum())



# indicator_df = pd.concat(  ) 

# indicator_df.index = text_name_list


3 # fundmentals 
data_income_df = pd.read_excel('src/net_income.xlsx')
# 这是临时的代码, 用于调整data_income_df的顺序和index
a = [ _ for _  in indicator_df.index.tolist() if _ not in ['RDSB', 'Tesco']]
data_income_df.index = data_income_df['Date'].values.tolist()
#data_income_df = data_income_df.fillna(method='pad')
data_income_df = data_income_df[a]
data_income_df = data_income_df.apply(lambda x: np.log(x/x.shift(1)),axis=0 )
data_income_df = data_income_df.iloc[1:,] 
data_income_df = data_income_df.T
data_income_df.fillna( data_income_df.mean(axis=0),inplace=True )



# data_income_df['intercept'] = 1
# 临时的代码



indicator_df = indicator_df.loc[a]

def calculate_metrics(series_X, series_Y):
    #series_X = data_income_df['2018Q1'].values
    temp_df = pd.DataFrame(series_X)
    temp_df.insert(0, 'intecpt',value=[1] * len(series_X))
    X = temp_df.values
    #X = indicator_df['LM_indicator']
    #Y = data_income_df['2018Q2']
    Y = series_Y
    sm.add_constant(X)
    result = sm.OLS(Y,X).fit()
    intercept = result.conf_int().values.tolist()[0][0]
    coef = result.conf_int().values.tolist()[0][1]
    p_value = result.pvalues.values[0]
    R2 = result.rsquared_adj
    
    #linear_reg = LinearRegression(fit_intercept=False)
    
    #linear_reg.fit(X, series_Y)
    #R2_linear = linear_reg.score(X, series_Y)

    return(pd.Series({
        'linear intercept': intercept,
        'linear Coef': coef,
        'p_value of coef': p_value,
        'linear_R2': R2
        
    }))

result_df_LM = data_income_df.apply(lambda x: calculate_metrics(x, indicator_df['LM_indicator']) ,axis=0)
result_df_GI = data_income_df.apply(lambda x: calculate_metrics(x, indicator_df['GI_indicator']) ,axis=0)
result_df_all = data_income_df.apply(lambda x: calculate_metrics(x, indicator_df['all_indicator']) ,axis=0)
result_df_FinBERT = data_income_df.apply(lambda x: calculate_metrics(x, indicator_df['FinBERT_indicator']) ,axis=0)


result_df_LM['Lexicon'] = 'LM'
result_df_GI['Lexicon'] = 'GI'
result_df_all['Lexicon'] = 'LM+GI+HW'
result_df_FinBERT['Lexicon'] = 'FinBERT'
r0 = pd.concat([result_df_LM, result_df_GI, result_df_all, result_df_FinBERT],axis=0)
r0.index = pd.MultiIndex.from_frame(r0.reset_index()[['Lexicon','index']])
r0 = r0.drop(['Lexicon'],axis=1)

# result 

#indicator_df.columns = [i +'_indicator' for i in indicator_df.columns]
r1 = pd.merge(indicator_df,data_income_df, left_index=True, right_index=True, how='outer')
r1.corr()

# X = indicator_df['LM_indicator']
# Y = data_income_df['2018Q2']
# sm.add_constant(X)
# result = sm.OLS(Y,X).fit()
# intercept = result.conf_int().values.tolist()[0][0]
# coef = result.conf_int().values.tolist()[0][1]
# p_value = result.pvalues.values[0]
# R2 = result.rsquared_adj

sns.set_theme(style="white")
# Generate a large random dataset

# Compute the correlation matrix
corr = r1.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))
# Set up the matplotlib figure
P_EDA_2 = plt.figure(figsize=(13,13))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
P_EDA_2 =sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)
P_EDA_2.set_title("the correlation heatmap")



r1.corr().iloc[-4:,]


r2 = r1.copy()

r2['GI_indicator_class'] = ( r2['GI_indicator'] < 0 ) 
#r2['GI_indicator_class'] = ( r2['HW_indicator'] < 0 ) 
r2['LM_indicator_class'] = ( r2['LM_indicator'] < 0 ) 
r2['all_indicator_class'] = ( r2['all_indicator'] < 0 ) 
r2['FinBERT_indicator_class'] = ( r2['FinBERT_indicator'] < 0 ) 


a = r2.groupby(['LM_indicator_class'])[['2018Q2','2018Q3','2018Q4','2019Q1']].apply(lambda x: x.mean())
a['Lexicon'] = 'LM'

b = r2.groupby(['GI_indicator_class'])[['2018Q2','2018Q3','2018Q4','2019Q1']].apply(lambda x: x.mean())
b['Lexicon'] = 'GI'

c = r2.groupby(['all_indicator_class'])[['2018Q2','2018Q3','2018Q4','2019Q1']].apply(lambda x: x.mean())
c['Lexicon'] = 'LM+GI+HW'

d = r2.groupby(['FinBERT_indicator_class'])[['2018Q2','2018Q3','2018Q4','2019Q1']].apply(lambda x: x.mean())
d['Lexicon'] = 'FinBERT'

r3 = pd.concat([a,b,c,d],axis=0)

r3.index = pd.MultiIndex.from_frame(r3.reset_index()[['Lexicon','index']])
r3 = r3.drop(['Lexicon'],axis=1)

# --- vadar ---  
# a = SentimentIntensityAnalyzer()
# sentiment_dict = a.polarity_scores(text)

# from nltk.stem.wordnet import WordNetLemmatizer

# -------------------------------


# each_list = text_list

# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
# model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# inputs = tokenizer(each_list, padding = True, truncation = True, return_tensors='pt')
# print(inputs)


# outputs = model(**inputs)
# print(outputs.logits.shape)


# import torch

# predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
# print(predictions)

# import pandas as pd

# positive = predictions[:, 0].tolist()
# negative = predictions[:, 1].tolist()
# neutral = predictions[:, 2].tolist()

# table = {'Company':pdf_name,
#          'Earning Call':each_list,
#          "Positive":positive,
#          "Negative":negative, 
#          "Neutral":neutral}
      
# df_finbert = pd.DataFrame(table, columns = ["Company","Earning Call", "Positive", "Negative", "Neutral"])


# prepare Finbert
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")


# pre-processing for NLP sentiment analysis model
# 注意： 这里的tokenizer 可以用list作为input

temp_list =[]
for i in range(len(text_list)):
    inputs = tokenizer(text_list[i], padding = True, truncation = True, return_tensors='pt')

    # get output 
    outputs = model(**inputs)
    print(outputs.logits.shape) 

    # max-soft
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    print(predictions)

    # visualization 
    positive = predictions[:, 0].tolist()
    negative = predictions[:, 1].tolist()
    neutral = predictions[:, 2].tolist()

    table = {'Headline':text_name_list[i],
            "Positive":positive,
            "Negative":negative, 
            "Neutral":neutral}
        
    df = pd.DataFrame(table, columns = ["Headline", "Positive", "Negative", "Neutral"])

    temp_list.append(df)

temp_df = pd.concat(temp_list)
temp_df['Positive'] - temp_df['Negative']



# # Lexicon 
# from lexicon import Lexicon




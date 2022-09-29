#%%

from calendar import EPOCH
from operator import index
from pickle import FALSE, TRUE
from pyexpat import model
from re import A
from turtle import Turtle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import optuna 
from pyod.models.auto_encoder import AutoEncoder
import seaborn as sns

import src 
import src.input.prepare_data 
import src.input.feature_engineer
import src.utilities.reportTools
import src.modelAD.model_selection
import src.modelAD.model_validation
import src.modelAD.model_evaluation

# ENVIRONMENT_FLAG = 'fastdev'
ENVIRONMENT_FLAG = 'dev'
TUNE_K_FLAG = False

#%%


#---------------------------------------------------
#------------------- Part 0: Params ------------------
# Set Params
PATH_ESG_FILE = './data/Eikon_ESG_2021_2014.csv'
PATH_MARKET_FILE = './data/Eikon_market_2022Q3_2013Q1.csv'
START_YEAR = 2017  # E 2016 year
END_YEAR = 2021
RANDOM_SEED = np.random.randint(1,1000,size = 1)[0] if ENVIRONMENT_FLAG in ('prod') else 33
 


#---------------------------------------------------
#------------------- Part 1: Data ------------------

# Read ESG raw data + Clean + drop ESG NA + 强行抹掉0 (drop NA & 0 since missing raw data )
df_ESG_Eikon = src.input.prepare_data.get_ESG_data(PATH_ESG_FILE, START_YEAR-1, END_YEAR)
df_ESG_Eikon = df_ESG_Eikon.dropna(subset=['ESG Score', 'Social Pillar Score', 'Governance Pillar Score']).fillna(0)        
df_ESG_Eikon = df_ESG_Eikon[df_ESG_Eikon['Environmental Pillar Score']!=0] 


# Read Market raw data + Clean
df_market_Eikon = src.input.prepare_data.get_market_data(PATH_MARKET_FILE)

# Generate risk features + Merge into df_all + drop NA since missing raw data 
df_risk_metrics = df_market_Eikon.groupby(['Symbol','Year']).apply(src.input.prepare_data.calculate_risk_metric_features)
df_all = pd.merge( df_ESG_Eikon, 
                   df_risk_metrics, 
                   how='left', 
                   left_index=True, 
                   right_index=True).sort_index().dropna() 

# Generate distance features (这里用0填充可能会产生问题)
df_distance = src.input.prepare_data.calculate_distance_features(df_all)
df_all = df_all[df_all.index.get_level_values(1)<=END_YEAR]
df_all = pd.merge( df_all[df_all.index.get_level_values(1)>=START_YEAR], 
                   df_distance, 
                   how='left', 
                   left_index=True, right_index=True)
                   

# Report ESG & Market data
if ENVIRONMENT_FLAG in ('pord', 'dev'):
    # src.utilities.reportTools.create_profile_report(df_ESG_Eikon, "./output/Report_ESG_data.html")
    # src.utilities.reportTools.create_profile_report(df_market_Eikon, "./output/Report_Markket_data.html")
    # src.utilities.reportTools.create_profile_report(df_risk_metrics, "./output/Report_metrics_data.html")
    src.utilities.reportTools.create_profile_report(df_all, "./output/Report_all_data_before_feature_engineer.html")


# #---------------------------------------------------
# #------------------- Part 2: Feature ---------------

# Data Transforamtion 
df_all = src.input.feature_engineer.transform_feature(df_all)

# Report all data after feature transforamtion
if ENVIRONMENT_FLAG in ('pord', 'dev'):
    src.utilities.reportTools.create_profile_report(df_all, "./output/Report_all_data_after_feature_transformation.html")


# TS Feature Extraction
df_input = src.input.feature_engineer.extract_feature(df_all)
df_input_raw = df_input.copy()
B = df_input.copy()
C = (B.isna().sum() <= B.shape[0] * 0.20)
df_input = B[ C[C == True].index.to_list()].copy()
# df_input = df_input.fillna(df_input.mean())

# Report all data after feature extraction
if ENVIRONMENT_FLAG in ('pord', 'dev'):
    
    A = df_input.copy()
    A.columns = [f'''{a}-{b}''' for a,b in zip(A.columns.get_level_values(0),A.columns.get_level_values(1))]
    src.utilities.reportTools.create_profile_report(A, "./output/Report_input_data_after_.html", minimal=True)


# Data Transformation after feature extraction
df_input = src.input.feature_engineer.transform_feature_after_extraction(df_input)



# B =src.input.feature_engineer.transform_feature_after_extraction(df_input)

# Impossible to open, do not open 
if ENVIRONMENT_FLAG in ('pord', 'dev'):
    A = df_input.iloc[:,:].copy()
    A.columns = [f'''{a}-{b}''' for a,b in zip(A.columns.get_level_values(0),A.columns.get_level_values(1))]
    src.utilities.reportTools.create_profile_report(A, "./output/Report_input_data.html", minimal=True)
    plt.figure()
    fig_corr_df_input = sns.heatmap(A.corr())

#%%

if TUNE_K_FLAG == True:
    # train FRUFS model
    model_frufs = src.modelAD.model_selection.build_FRUST_model(df=df_input, random_state=RANDOM_SEED)

    # Plot Feature Importance
    fig_fs_importance = src.modelAD.model_selection.plot_feature_importance(model_frufs) # SLOW


    fig_tune_K, df_fs_record, model_best_list, K_best, df_input_pruned =src.modelAD.model_selection.plot_feature_selection(
        model_frufs=model_frufs, 
        df_input=df_input, 
        n_trails=15
    ) # SLOW

    # update fig_fs_importance
    model_frufs_new = src.modelAD.model_selection.build_FRUST_model(df=df_input, random_state=RANDOM_SEED, k=K_best)
    fig_fs_importance = src.modelAD.model_selection.plot_feature_importance(model_frufs_new)
else:
    K_best = 14
    df_input_pruned = df_input[[(               'mean_change',       'semi-variance (down)'),
            (               'mean_change', 'Environmental Pillar Score'),
            (               'mean_change',                  'VaR (95%)'),
            (               'mean_change',                  'ESG Score'),
            (               'mean_change',    'Governance Pillar Score'),
            (               'mean_change',        'Social Pillar Score'),
            ('Environmental Pillar Score',                       2020.0),
            (                 'VaR (95%)',                       2020.0),
            (               'mean-return',                       2020.0),
            (       'Social Pillar Score',                       2020.0),
            (   'Governance Pillar Score',                       2020.0),
            (                       'std',                  'VaR (95%)'),
            (                 'ESG Score',                       2020.0),
            (      'semi-variance (down)',                       2020.0)]]
    

#%%



# Best K = 14:
# MultiIndex([(               'mean_change',       'semi-variance (down)'),
#             (               'mean_change', 'Environmental Pillar Score'),
#             (               'mean_change',                  'VaR (95%)'),
#             (               'mean_change',                  'ESG Score'),
#             (               'mean_change',    'Governance Pillar Score'),
#             (               'mean_change',        'Social Pillar Score'),
#             ('Environmental Pillar Score',                       2020.0),
#             (                 'VaR (95%)',                       2020.0),
#             (               'mean-return',                       2020.0),
#             (       'Social Pillar Score',                       2020.0),
#             (   'Governance Pillar Score',                       2020.0),
#             (                       'std',                  'VaR (95%)'),
#             (                 'ESG Score',                       2020.0),
#             (      'semi-variance (down)',                       2020.0)],
#            )







# #---------------------------------------------------
# #------------------- Part 3: Model: Train, Tune, Evaluation ---------------

# split dataset 
df_train = df_input_pruned.copy()
df_train = df_input_pruned[(df_input_pruned != 0).sum(axis=1) == K_best] 
X_train = df_train.values.copy() # Unsupervised, no need for testing dataset

# train & tune model 
study, model_best = src.modelAD.model_selection._train_model(X_train, n_trails=35)
fig_tune = optuna.visualization.plot_optimization_history(study)


# Evaluate Model 
# Plot (1) - Loss graph 
# Plot The Loss Function; Problem: where is the validation dataset? 
fig_loss = src.modelAD.model_evaluation.plot_loss_history( history= model_best.history_)

# # Plot (2) - t-SNE 
fig_tSNE = src.modelAD.model_evaluation.plot_t_SNE(X_train, model_best.labels_)



# Prepare data for explaining
df_results = df_train.copy()
df_results['Anomaly_Score'] = model_best.decision_function(X_train)
df_results['Proba (is A)'] = model_best.predict_proba(X_train)[:,1]
df_results['Label'] = model_best.labels_

df_all['semi-variance (down)'] = df_all['semi-variance (down)'].apply(np.abs)
# find out best & worst 
df_results = df_results.sort_values(('Proba (is A)',''),ascending= False)
df_results = df_results[df_results.index != 'QRVO.OQ']

# #---------------------------------------------------
# #------------------- Part 3: Model: Explaination -----------------
# feature_distribution 
fig_group_distribution, df_group_distribution = src.modelAD.model_validation.plot_feature_distribution(df=df_results)

## Investigation
# Anomaly
fig_anomaly_high_ts, df_anomaly_high_ts= src.modelAD.model_validation.plot_top_K_ts(top_K=4, df_results=df_results, df_all=df_all)

# Normal
fig_anomaly_low_ts, df_anomaly_low_ts= src.modelAD.model_validation.plot_top_K_ts(top_K=-4, df_results=df_results, df_all=df_all)


#%%

# SHAP (SLOW)
# explain model
import shap
explainer = shap.KernelExplainer(model_best.predict, X_train)
shap_values = explainer.shap_values(X_train, nsamples = 200 )
    
fig_1 = shap.summary_plot(shap_values, X_train)
fig_2 = shap.summary_plot(shap_values, X_train, plot_type="bar")


# return( (model_NN, r2, explainer, shap_values, [fig_1, fig_2]) )


import datapane as dp
report = dp.Report(
            dp.Page(title="data",
                blocks=[
                    dp.Text("### Corr of input data"),
                    dp.Plot(fig_corr_df_input),
                ]
            ),
            dp.Page(title="Feature",
                blocks=[
                    dp.Text("### The figure of feature importance"),
                    dp.Plot(fig_fs_importance),
                    dp.Text("### The figure of tuning K"),
                    dp.Plot(fig_tune_K),
                    dp.Text("### The recording of tuning K"),
                    dp.Plot(df_fs_record),
                ]
            ),
            dp.Page(title="Model Train & Tune & Evaluation",
                blocks=[
                    dp.Text("### The structure of Model"),
                    dp.Text("Please check"),
                    # dp.Plot(src.modelAD.model_selection.plot_model(model_best) ),
                    dp.Text("### The figure of Tuning"),
                    dp.Plot(fig_tune),
                    dp.Text("### The figure of Loss"), 
                    dp.Plot(src.modelAD.model_evaluation.plot_loss_history( history= model_best.history_)),
                    dp.Text("### The figure of t-SNE"), 
                    dp.Plot(fig_tSNE),
                ]
            ),
            dp.Page(title="Model Validation", 
                blocks=[
                    dp.Text("### The reuslts of AD"), 
                    # dp.DataTable(df_results), 
                    dp.Text("### The figure & data of distribution"), 
                    fig_group_distribution,
                    # dp.Text("### The figure of distribution"), 
                    dp.DataTable(df_group_distribution), 
                    dp.Text("### The figure & data of anomaly time-series"), 
                    fig_anomaly_high_ts,
                    dp.DataTable(df_anomaly_high_ts ), 
                    dp.Text("### The figure & data of normal tiem-series"), 
                    fig_anomaly_low_ts,
                    dp.DataTable(df_anomaly_low_ts), 
                ]
            ),
            # dp.Page(title="Titanic Dataset", blocks=["### Dataset", titanic]),
        )
report.save(path="./output/Report.html")
# %%




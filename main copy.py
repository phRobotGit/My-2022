#%%

from calendar import EPOCH
from operator import index
from pyexpat import model
from re import A
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import src 
import src.input.prepare_data 
import src.input.feature_engineer
import src.utilities.reportTools
import src.modelAD.model_selection
import src.modelAD.model_validation
import src.modelAD.model_evaluation

# ENVIRONMENT_FLAG = 'fastdev'
ENVIRONMENT_FLAG = 'dev'
#%%
#---------------------------------------------------
#------------------- Part 0: Params ------------------

# Set Params
PATH_ESG_FILE = './data/Eikon_ESG_2021_2014.csv'
PATH_MARKET_FILE = './data/Eikon_market_2022Q3_2013Q1.csv'
START_YEAR = 2016  # E 2016 year
 
#---------------------------------------------------
#------------------- Part 1: Data ------------------

# Read ESG raw data + Clean + drop ESG NA
df_ESG_Eikon = src.input.prepare_data.get_ESG_data(
                    PATH_ESG_FILE, 
                    START_YEAR-1).dropna(
                        subset=['ESG Score', 'Social Pillar Score', 'Governance Pillar Score']
                    ).fillna(0)

# 强行把0抹去，试试结果                    
df_ESG_Eikon = df_ESG_Eikon[df_ESG_Eikon['Environmental Pillar Score']!=0]
# Drop NA, since missing raw data 

# Read Market raw data + Clean
df_market_Eikon = src.input.prepare_data.get_market_data(PATH_MARKET_FILE)

# Generate risk features 
df_risk_metrics = df_market_Eikon.groupby(['Symbol','Year']).apply(
                        src.input.prepare_data.calculate_risk_metric_features
                        )

df_all = pd.merge( df_ESG_Eikon, 
                   df_risk_metrics, 
                   how='left', 
                   left_index=True, 
                   right_index=True).sort_index().dropna() 
# Drop NA, since missing raw data

# Generate distance features 
df_distance = src.input.prepare_data.calculate_distance_features(df_all).fillna(0)
# 这里用0填充可能有问题 ？？？！！！

df_all = pd.merge( df_all[df_all.index.get_level_values(1)>=START_YEAR], 
                   df_distance, 
                   how='left', 
                   left_index=True, right_index=True)


# Report ESG & Market data
# if ENVIRONMENT_FLAG in ('pord', 'dev'):
    # src.utilities.reportTools.create_profile_report(df_ESG_Eikon, "./output/Report_ESG_data.html")
    # src.utilities.reportTools.create_profile_report(df_market_Eikon, "./output/Report_Markket_data.html")
    # src.utilities.reportTools.create_profile_report(df_risk_metrics, "./output/Report_metrics_data.html")
    # src.utilities.reportTools.create_profile_report(df_all, "./output/Report_all_data.html")

# 




# #---------------------------------------------------
# #------------------- Part 2: Feature ---------------

# Data Transforamtion 
df_all['semi-variance (down)'] = df_all['semi-variance (down)'].apply(np.log)
df_all['VaR (95%)'] = df_all['VaR (95%)'].apply(lambda x: np.sign(x) * np.power( np.abs(x), 1/3))
# df_all['D(Overall,E)'] = df_all['D(Overall,E)'].apply(lambda x: np.power(x, 1/3))
# df_all['D(Overall,S)'] = df_all['D(Overall,S)'].apply(lambda x: np.power(x, 1/3))
# df_all['D(Overall,G)'] = df_all['D(Overall,G)'].apply(lambda x: np.power(x, 1/3))
df_all['D(ESG, VaR)'] = df_all['D(ESG, VaR)'].apply(lambda x: np.power(x, 1/3))

if ENVIRONMENT_FLAG in ('pord', 'dev'):
    src.utilities.reportTools.create_profile_report(df_all, "./output/Report_new_all_data.html")

#%%

# TS Feature Extraction

df_input = df_all.groupby(by = df_all.index.get_level_values(0) ).apply(src.input.feature_engineer.extract_ts_feature)
df_all_flat = df_all.unstack(level=1)
df_all_flat = df_all_flat.fillna(df_all_flat.mean())

df_input = pd.merge(df_input, df_all_flat, how='left', 
                    left_index=True, right_index=True)


if ENVIRONMENT_FLAG in ('pord', 'dev'):
    A = df_input.copy()
    A.columns = [f'''{a}-{b}''' for a,b in zip(A.columns.get_level_values(0),A.columns.get_level_values(1))]
    src.utilities.reportTools.create_profile_report(A, "./output/Report_input_data.html", minimal=True)


#%%
# 强行处理 missing value 回头有空了再来填补这里的空白
df_input = df_input.fillna(df_input.mean(axis=0))
df_input = df_input.dropna(axis=0)

# Standardize Transformation
from sklearn.preprocessing import StandardScaler
df_input = pd.DataFrame(
                StandardScaler().fit_transform(df_input),
                columns= df_input.columns,
                index=df_input.index)

# Feature Selcetion 
from FRUFS import FRUFS
from lightgbm import LGBMClassifier, LGBMRegressor
# Initialize the FRUFS object


model_frufs = FRUFS(
                model_r=LGBMRegressor(random_state=27), 
                model_c=LGBMClassifier(random_state=27, class_weight="balanced"), 
                # categorical_features=categorical_features, 
                k=50, n_jobs=-1, verbose=0, 
                random_state=27)

df_input_pruned = model_frufs.fit_transform(df_input)


# plt.figure(figsize=(15,25),dpi=100)
fig_fs_importance = src.modelAD.model_selection.plot_feature_importance(model_frufs)

#%%

# Impossible to open, do not open 
if ENVIRONMENT_FLAG in ('pord', 'dev'):
    A = df_input.iloc[:,:].copy()
    A.columns = [f'''{a}-{b}''' for a,b in zip(A.columns.get_level_values(0),A.columns.get_level_values(1))]
    src.utilities.reportTools.create_profile_report(A, "./output/Report_input_data.html", minimal=True)


# #---------------------------------------------------
# #------------------- Part 3: Model: Train, Tune, Evaluation ---------------

import optuna 
from pyod.models.auto_encoder import AutoEncoder

# split dataset 
df_train = df_input_pruned.copy() 
X_train = df_train.values.copy() # Unsupervised, no need for testing dataset

# tune the optimal K in feature selection 

# train & tune model 
study, model_best = src.modelAD.model_selection.train_model(X_train)
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

# find out best & worst 
df_results = df_results.sort_values(('Proba (is A)',''),ascending= False)



# #---------------------------------------------------
# #------------------- Part 3: Model: Explaination -----------------
# feature_distribution 
fig_group_distribution, df_group_distribution = src.modelAD.model_validation.plot_feature_distribution(df=df_results)

## Investigation
# Anomaly
fig_anomaly_high_ts, df_anomaly_high_ts= src.modelAD.model_validation.plot_top_K_ts(top_K=4, df_results=df_results, df_all=df_all)

# Normal
fig_anomaly_low_ts, df_anomaly_low_ts= src.modelAD.model_validation.plot_top_K_ts(top_K=-4, df_results=df_results, df_all=df_all)


import datapane as dp
report = dp.Report(
            dp.Page(title="Feature",
                blocks=[
                    dp.Text("### The figure of "),
                    dp.Plot(fig_fs_importance),
                ]
            ),
            dp.Page(title="Model Train & Tune & Evaluation",
                blocks=[
                    dp.Text("### Model"),
                    # dp.Plot(src.modelAD.model_selection.plot_model(model_best) ),
                    dp.Text("### The figure of Tune"),
                    dp.Plot(fig_tune),
                    dp.Text("### The figure of Loss"), 
                    dp.Plot(src.modelAD.model_evaluation.plot_loss_history( history= model_best.history_)),
                    dp.Text("### The figure of t-SNE"), 
                    dp.Plot(fig_tSNE),
                ]
            ),
            dp.Page(title="Model Validation", 
                blocks=[
                    dp.Text("### The figure of distribution"), 
                    fig_group_distribution,
                    dp.Text("### The figure of distribution"), 
                    # dp.DataTable(df_group_distribution), 
                    dp.Text("### The figure of anomaly time-series"), 
                    fig_anomaly_high_ts,
                    # dp.Table(),
                    dp.Text("### The figure of normal tiem-series"), 
                    fig_anomaly_low_ts,
                ]
            ),
            # dp.Page(title="Titanic Dataset", blocks=["### Dataset", titanic]),
        )
report.save(path="./output/Report.html")
# %%

# df_group_distribution



symbol_list =df_input.index.to_list()
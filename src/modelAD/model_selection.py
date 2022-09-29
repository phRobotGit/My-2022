from pyexpat import model
from symbol import yield_arg
import tensorflow as tf
import random as rn
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from pyod.models.auto_encoder import AutoEncoder
import optuna
from keras.utils.vis_utils import model_to_dot 
from IPython.display import SVG
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import seaborn as sns 
from FRUFS import FRUFS
from lightgbm import LGBMClassifier, LGBMRegressor
import optuna 
from pyod.models.auto_encoder import AutoEncoder


def optuna_AutoEncoder(trial, X_train):
    n,m = X_train.shape
    layer_c0_node_num = trial.suggest_int("layer_1_node_num", 1,int(0.3*m),3)
    layer_c1_node_num = trial.suggest_int("layer_2_node_num", 1,int(0.7*m),5)
    layer_c2_node_num = trial.suggest_int("layer_3_node_num", 1,m,10)
    # layer_c0_node_num = trial.suggest_int("layer_1_node_num", 1,21,3)
    # layer_c1_node_num = trial.suggest_int("layer_2_node_num", 1,51,5)
    # layer_c2_node_num = trial.suggest_int("layer_3_node_num", 1,51,5)
    
    model = AutoEncoder(
        hidden_neurons=[
            layer_c2_node_num,
            layer_c1_node_num,
            layer_c0_node_num,
            layer_c1_node_num,
            layer_c2_node_num
        ], 
        loss='mean_squared_error',
        dropout_rate= trial.suggest_float(name="drop_out_rate", low=0.1, high=0.4, step=0.1),
        validation_size=0.1, 
        contamination=0.01, 
        verbose=0
    )
    return(model)

def object(trial, X_train):
    model = optuna_AutoEncoder(trial, X_train)
    model.fit(X_train)
    loss = model.history_["loss"][-1]
    return( loss )


def optuna_get_trial_detail(trial, X_train):
    '''
        get detials 
    '''
    model = optuna_AutoEncoder(trial, X_train)
    model.fit(X_train)
    return(
        model
    )


def _train_model(X_train, n_trails=35):
    study = optuna.create_study(direction="minimize")
    study.optimize(
                    lambda trial: object(trial, X_train), 
                    n_trials=n_trails
                )
    model_best = optuna_get_trial_detail(study.best_trial, X_train)
    return(study, model_best)



def plot_model(model):
    return(
        SVG(model_to_dot(
            model.model_, 
            show_shapes=True, 
            show_layer_names=True, 
            rankdir='TB').create(prog='dot', format='svg'))
    )

def plot_feature_importance(model):
    y_aixs = np.arange( len(model.columns_) )
    x_aixs = model.feat_imps_

    fig = plt.figure(figsize=(15,25))
    sns.lineplot( x = x_aixs, y=[model.k for i in range(len(y_aixs))], linestyle='--' )
    sns.barplot( x=x_aixs, y=y_aixs, orient='h')
    if type(model.columns_[0]) ==str:
        plt.yticks(y_aixs, model.columns_, size='small')
    else:
        plt.yticks(y_aixs, ["Feathre"+str(i) for i in model.columns_], size='small')
    
    for x, y in zip(x_aixs, y_aixs):   # # 添加数据标签
        plt.text(x+3, y, f'''{x:.2f}''', ha='center', va='bottom',rotation=0)    
    
    plt.xlabel("Imporance Scores")
    plt.ylabel("Features")
    sns.despine()
    return(fig)


# .....
def _plot_fig_tune_K(df_fs_record):
    fig = plt.figure(figsize=(15,10))
    x_aixs = df_fs_record.index.to_list()
    y_aixs = df_fs_record['val_loss'].tolist()
    y_val_loss = df_fs_record['benchmark (val)'].values[0]
    y_shift = 0.01

    sns.lineplot(data=df_fs_record)

    for x, y in zip(x_aixs, y_aixs):   # # 添加数据标签
        plt.text(x, y+y_shift, f'''{y:.2f}''')    

    plt.text(x_aixs[-1], y_val_loss +y_shift, f'''{y_val_loss:.2f}''')
    plt.xlabel("K")
    plt.ylabel("Loss")
    return(fig)



def plot_feature_selection(model_frufs, df_input, n_trails=15):

    # prepare 
    feat_imp_dict = dict(zip(model_frufs.columns_, model_frufs.feat_imps_))
    feats = list( feat_imp_dict.keys() )
    # K_trial_list = [0.05*i for i in range(1,4)]
    # K_trial_list = [0.05*i for i in range(1,20)]
    K_trial_list = [0.03*i for i in range(15,30)]

    loss_list = []
    df_input_pruned_list = []
    model_best_list = []
    study_list = []

    for K_trial in tqdm.tqdm(K_trial_list):
        
        # prepare df_input_pruned 
        use_feats = feats[:int(len(feats)*K_trial)] 
        df_input_pruned = df_input[use_feats].copy()
        df_input_pruned_list.append(df_input_pruned)

        # train & tune model 
        df_train = df_input_pruned.copy() 
        X_train = df_train.values.copy() # Unsupervised, no need for testing dataset
        study, model_best = _train_model(X_train, n_trails=n_trails)
        
        study_list.append(study)
        model_best_list.append( model_best)
        loss_list.append( pd.DataFrame(model_best.history_).iloc[-1,:] )


    bc_study, bc_model_best = _train_model(df_input.values, n_trails=n_trails)
    bc_loss = pd.DataFrame(bc_model_best.history_)['loss'].iloc[-1]
    bc_val = pd.DataFrame(bc_model_best.history_)['val_loss'].iloc[-1]    


    # Plot
    df = pd.concat(loss_list, axis=1)
    df.columns = [ int(len(feats)*i) for i in K_trial_list]
    # fig_tune_K = plt.figure()
    df = df.T
    df['benchmark (loss)'] = bc_loss
    df['benchmark (val)'] = bc_val
    
    
    
    #df.plot.line()
    # df.plot.line(x=df.T.index.to_list(), y=[1]*df.shape[0])
    # sns.lineplot()
    # plt.xlabel("K")
    # plt.ylabel("Loss")

    df_fs_record = df.copy()

    fig_tune_K = _plot_fig_tune_K(df_fs_record)

    # best K 
    K_best = df_fs_record.index[df_fs_record['val_loss'].argmin()]
    df_input_pruned = df_input_pruned = df_input[ feats[:K_best] ]
    return( fig_tune_K, df_fs_record, model_best_list, K_best, df_input_pruned)


def build_FRUST_model(df, random_state, k=70):
    df_input = df.copy()
    
    # Initialize the FRUFS object
    model_frufs = FRUFS(
                    model_r=LGBMRegressor(random_state=random_state), 
                    model_c=LGBMClassifier(class_weight="balanced"), 
                    # categorical_features=categorical_features, 
                    k=k, n_jobs=-1, verbose=0, 
                )

    model_frufs.fit(df_input)

    return(model_frufs)



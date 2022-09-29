# # Plot (2) - t-SNE 
import altair as alt 
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_loss_history(history):
    return(
        pd.DataFrame.from_dict(history).plot(title='Error Loss History')
    )

    # # history 
    # fig =plt.figure()
    # plt.plot( range(1,len(loss_train)+1), 
    #       loss_train,
    #       'g',
    #       label='Training loss')
    # plt.title("Training loss")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.legend()
    # return( fig )





def plot_t_SNE(X, labels):
    label = [ 'anomaly' if i==1 else 'normal' for i in labels]

    X_TSNE = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30).fit_transform(X)
    df_X_TSNE = pd.DataFrame( X_TSNE, columns=['Dim-1','Dim-2'] )
    df_X_TSNE['label'] = [ f"Group {s}" for s in label]

    fig = alt.Chart(df_X_TSNE).mark_circle(size=60).encode(
        x='Dim-1',
        y='Dim-2',
        color = 'label'
    ).interactive()

    return(fig)
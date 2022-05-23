from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt 
import statsmodels as sm
from statsmodels.tsa import stattools
import pandas as pd
import numpy as np

def plot_acf_pacf(x_series, x_variable_name=None):
    fig, ax = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0.5)

    plot_acf(x_series, ax=ax[0])
    ax[0].set_title(f'''ACF({x_variable_name})''')
    plot_pacf(x_series, ax=ax[1])
    ax[1].set_title(f'''PACF({x_variable_name})''')

    plt.show()
    return(fig)


def test_autocorrelation(x_series):
    at2 = x_series
    m = 25 # 我们检验25个自相关系数
    acf,q,p = stattools.acf(at2,nlags=m,qstat=True)  ## 计算自相关系数 及p-value
    out = np.c_[range(1,26), acf[1:], q, p]
    output=pd.DataFrame(out, columns=['lag', "AC", "Q", "P-value"])
    output = output.set_index('lag')
    return(output)
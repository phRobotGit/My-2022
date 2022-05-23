import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

def plot_uni_charts(df, title = '', column_num= 3):
    a = df 
    P_EDA_1 = alt.Chart(a.reset_index()).transform_fold(
        a.columns.tolist(),
        as_ = ['name', 'value']
    ).mark_line(tooltip=alt.TooltipContent('encoding')).encode(
        x = 'Date:T',
        y = 'value:Q',
        color = 'name:N',
    ).facet(
        facet='name:N',
        columns=column_num
    ).resolve_scale(
        x='independent', 
        y='independent'
    ).properties(
        title= title
    )
    return(P_EDA_1)

import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
import re 
from plotly import subplots

# table
def plot(metrics_by_trader, metrics_by_trader_date, metrics_by_trader_sector, trader, date ):
    
    #(1) Table  
    table_metris = ff.create_table(metrics_by_trader[['trader', 'return', 'deviation','net_amount']], height_constant=50)
    table_metris.write_html('src/table_metris.html')


    #(2) bubble
    bubble = px.scatter(metrics_by_trader, x="deviation", y="return", color="trader",
                 size='net_amount')


    #(3) line
    df = metrics_by_trader_date[ (metrics_by_trader_date['trader']==trader) &
                             (metrics_by_trader_date['cob_date']<=date)].copy()
    line =  px.line(df, x='cob_date', y='net_amount',color='trader', symbol="trader")

 
    # pie    
    df = metrics_by_trader_sector[metrics_by_trader_date['trader']==trader].copy()
    labels = df['GICSSector'].to_list()
    values = df['net_amount'] / df['net_amount'].sum()
    pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])


    # bar_return & bar_deviation
    df = metrics_by_trader_sector[metrics_by_trader_date['trader']==trader].copy()
    bar_deviation = px.bar(df, x='GICSSector', y='deviation')   
    bar_return = px.bar(df, x='GICSSector', y='return')   

    return(table_metris, bubble, line, pie, bar_deviation, bar_return )



# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 22:33:34 2021

@author: Peanut Robot
"""
import numpy as np
import dash 
import dash_core_components
import dash_html_components
from pymongo import MongoClient
import datetime 

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


def dash_plot(): #data_trades
###(3) MongoDB connect
    con = MongoClient('mongodb://localhost')    
    data_mongo = pd.DataFrame( con.db.CourseworkTwo.find({}) ) # key:  TradeID (DateTime + Symbol -- Counterparty)   # 交易信息
    con.close()
    
    data_mongo['Price'] = data_mongo['Notional'] / data_mongo['Quantity']
    data_mongo["TimeStamp"] = data_mongo['DateTime'].apply(lambda x: datetime.datetime.strptime(x, "ISODate(%Y-%m-%dT%H:%M:%S.000Z)") ).values
    data_mongo['Date'] = data_mongo['TimeStamp'].apply( lambda x: x.date() )
    data_mongo['Time'] = data_mongo['TimeStamp'].apply( lambda x: x.time() )


    x_variable_list = ['Date','Time']
    y_variable_list = ['Quantity','Notional','Price']
    group_varible_list = ['Trader','Symbol','Counterparty','TradeType']

    date_series = data_mongo.Date.sort_values().unique()
    date_series_pos = range(len(date_series))
    date_series_str = [i.strftime("%Y-%m-%d") for i in date_series]
    #date_series_str = [i for i in date_series]

    app = JupyterDash(__name__)


    app.layout = html.Div(children=[
        html.H1(children="Test Graph"),
        html.H3("Choose x_variable"),
        dcc.Dropdown(id="x_variable",
                     options=[{"label":i,"value":i} for i in x_variable_list],
                     value = "Date"),
        html.H3("Choose y_variable"),
        dcc.Dropdown(id="y_variable",
                     options=[{"label":i,"value":i} for i in y_variable_list],
                     value = "Quantity"),
        html.H3("Choose group_variable"),
        dcc.Dropdown(id="group_variable",
                     options=[{"label":i,"value":i} for i in group_varible_list],
                     value = "Trader"),
        dcc.Graph(id='Time-varible'),
        
        dcc.RangeSlider(id='year_range',
                        min=min(date_series_pos),
                        max=max(date_series_pos),
                        value=[min(date_series_pos), max(date_series_pos)],
                        marks={ k:{'label':v}for k,v in zip(date_series_pos, date_series_str)},
                        allowCross=False,
                        tooltip={"placement": "bottom", "always_visible": True},
                        pushable=1
                        ),
        # -------------
        html.Div(children=[
            html.H2('Subplot'),
            html.H3("Choose x_plot2"),
            dcc.Dropdown(id="x_plot2",
                         options=[{"label":i,"value":i} for i in x_variable_list],
                         value = "Date"),
            html.H3("Choose y_plot2"),
            dcc.Dropdown(id="y_plot2",
                         options=[{"label":i,"value":i} for i in y_variable_list],
                         value = "Quantity"),
            html.H3("Choose group_plot2"),
            dcc.Dropdown(id="group_plot2",
                         options=[{"label":i,"value":i} for i in group_varible_list],
                         value = "Trader"),
            html.H3("Choose conditions on y variable"),
            dcc.Dropdown(id="condition_var",
                         options=[{"label":i,"value":i} for i in group_varible_list],
                         value = "Trader"),
            html.H4("Symbol == ? "),
            dcc.Checklist(id="condition1",
                          options=[ {'label': i, 'value': i} for i in data_mongo.Symbol.unique()],
                          value=['IQV'],
                          labelStyle={'display': 'inline-block'}
                          ),
            html.H4("Trader == ? "),
            dcc.Checklist(id="condition2",
                          options=[ {'label': i, 'value': i} for i in data_mongo.Trader.unique()],
                          value=[],
                          labelStyle={'display': 'inline-block'}
                          ),
            html.H4("CounterParty == ? "),
            dcc.Checklist(id="condition3",
                          options=[ {'label': i, 'value': i} for i in data_mongo.Counterparty.unique()],
                          value=[],
                          labelStyle={'display': 'inline-block'}
                          ),
            dcc.Graph(id='subplot')
            ])
        ])

    @app.callback(
        Output('Time-varible','figure'),
        [Input('x_variable','value'),
         Input('y_variable','value'),
         Input('group_variable','value'),
         Input('year_range','value')
         ])
    def time_varible_figure(x_variable, y_variable, group_variable, year_range):
        df = data_mongo.copy()
        df = df.sort_values(by=x_variable)
        df = df[df.Date >= date_series[year_range[0]] ]
        df = df[df.Date <= date_series[year_range[1]] ]
        fig = px.scatter(df,
                         x=x_variable,
                         y=y_variable,
                         color=group_variable,
                         symbol=group_variable
                         )
        fig.update_layout(transition_duration=500)
        return fig





    @app.callback(
        Output('subplot','figure'),
        [Input('x_plot2','value'),
         Input('y_plot2','value'),
         Input('group_plot2','value'),
         Input('condition_var','value'),
         Input('condition1','value'),
         Input('condition2','value'),
         Input('condition3','value'),
         Input('year_range','value')
         ])
    def subplot(x_plot2, y_plot2, group_plot2, condition_var, condition1, condition2, condition3, year_range):
        df = data_mongo.copy()
        df = df.sort_values(by=x_plot2)
        df = df[df.Date >= date_series[year_range[0]] ]
        df = df[df.Date <= date_series[year_range[1]] ]
        if condition_var == 'Symbol':
            df = df[df['Symbol'].apply( lambda x: x in condition1) ]
        elif condition_var == 'Trader':
            df = df[df['Trader'].apply( lambda x: x in condition2) ]
        elif condition_var == 'Counterparty':
            df = df[df['Counterparty'].apply( lambda x: x in condition3) ]
        fig = px.scatter(df,
                             x=x_plot2,
                             y=y_plot2,
                             color=group_plot2,
                             symbol=group_plot2
                                    )
        fig.update_layout(transition_duration=500)
        return fig

    app.run_server(mode='external',  port = 8052, dev_tools_ui=True, debug=True,
              dev_tools_hot_reload =True, threaded=True)  # external,inline
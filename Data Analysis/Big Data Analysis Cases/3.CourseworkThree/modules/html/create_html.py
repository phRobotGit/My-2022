
import pandas as pd 
import numpy as np 


def create_home_html(table_metris_html, bubble_html, line_html, pie_html, bar_deviation_html, bar_return_html, trader, date):
    date = pd.to_datetime(date).strftime('%d-%b-%Y')
    html = f'''
    <html>
    
        <head>
            <title>CW3: Report of six portfolios</title>
        </head>
        <body>
            <h1>CW3: Report of six portfolios</h1>
            <p></p>

            <br></br>
            <br></br>

            <h2>Part1:Overview of the six portfolios</h2>
            <p>Introduction: This part shows:</p>
            <p> (1) Main metrics of the six portfolios <p>
            <p> (2) The bubble chart of the six portfolios <p>
            <br></br>
            <h2>Metrics</h2>
            <p>The metrics calcalated based on the window from 15-Oct-2021 to {date} </p>
            <p>Tips:</p>
            <p>return: The holding contineous return during the period.</p>
            <p>deviation: The deviation of the daily contineous return of the portfolio </p>
            <p>net_amount: Net Amonut of the portfolio.</p>
            <div>
                { table_metris_html }
            </div>

            <h2>Bubble chart: The return, deviation, and net amount of each portfolio</h3>
            <p>Tips: Y-axis: return;  X-axis: deviation; Size: net amount</p>
            <div>
                { bubble_html }
            </div>
            <br></br>


            <h2>Part2:Deeper Analysitics of {trader} (from 15-Oct-2021 to {date}) </h2>
            <p> According to the bubble plot and the table, we can find that {trader} has the biggest deviation. So we we investigate it further<p>
            <p> we will use: </p> 
            <p> (1) the line chart to inspect its trendcy of the net amount </p>
            <p> (2) the pie plot to observe its allocation among GICSSecror </p> 
            <br></br>

            <h2>Line Chart: The net amount of {trader}</h2>
            <p> According to the line chart, we can see significant volatility in the net ammount</p>
            <div>
                { line_html }
            </div>

            <h3>Pie Chart: The proportion of net amount in different sectors</h3>
            <p>The pie chart shows that Information Technology dominates the proportion of the portfolio<p>
            <p>We analysis the most important contributor to the performance & risk by ploting bar_return & bar_deviation<p>
            <div>
                { pie_html }
            </div>
            <h3>Bar Chart: The holding-period returns in different sectors</h3>
            <div>
                { bar_return_html }
            </div>
            <h3>Bar Chart: The deviation of the holding-period returns in different sectors</h3>
            <div>
                { bar_deviation_html }
            </div>
        </body>
    </html>
    '''
    with open('home.html', 'w') as f:
        f.write(html)





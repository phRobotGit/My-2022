// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = 'Nunito', '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#858796';

const data = {{
  datasets: [{{
    label: 'First Dataset',
    data: [{{
      x: 20,
      y: 30,
      r: 80,
      c:'sa'
    }}, {{
      x: 40,
      y: 10,
      r: 10,
      c:'be'
    }}],
    backgroundColor: 'rgb(255, 99, 132)'
  }}]
}};

const config = {{
  type: 'bubble',
  data: data,
  options: {{}}
}};


// Bubble Chart Example
var ctx = document.getElementById("myBubbleChart");
var myBubbleChart = new Chart(ctx, {{
  type: 'bubble',
  data: data,
  options: {{
    maintainAspectRatio: false,
    layout: {{
      padding: {{
        left: 10,
        right: 25,
        top: 25,
        bottom: 0
      }}
    }},
    scales: {{
      xAxes: [{{
        time: {{
          unit: 'month'
        }},
        gridLines: {{
          display: false,
          drawBorder: false
        }},
        ticks: {{
          maxTicksLimit: 6
        }},
        maxBarThickness: 25,
      }}],
      yAxes: [{{
        ticks: {{
          min: 0,
          max: 100,
          maxTicksLimit: 5,
          padding: 10,
          // Include a dollar sign in the ticks
          callback: function(value, index, values) {{
            return '$' + number_format(value);
          }}
        }},
        gridLines: {{
          color: "rgb(234, 236, 244)",
          zeroLineColor: "rgb(234, 236, 244)",
          drawBorder: false,
          borderDash: [2],
          zeroLineBorderDash: [2]
        }}
      }}],
    }},
    legend: {{
      display: false
    }},
    tooltips: {{
      titleMarginBottom: 10,
      titleFontColor: '#6e707e',
      titleFontSize: 14,
      backgroundColor: "rgb(255,255,255)",
      bodyFontColor: "#858796",
      borderColor: '#dddfeb',
      borderWidth: 1,
      xPadding: 15,
      yPadding: 15,
      displayColors: false,
      caretPadding: 10,
      callbacks: {{
        label: function(tooltipItem, chart) {{
          var datasetLabel = chart.datasets[tooltipItem.datasetIndex].label || '';
          var radius = chart.datasets[0].data[tooltipItem.index].r;
          var c = chart.datasets[0].data[tooltipItem.index].c;
          return [datasetLabel + ': $' + number_format(tooltipItem.yLabel)  + radius+c,
                  'sss',
                  'bbb'];
        }}
      }}
    }},
  }}
}});

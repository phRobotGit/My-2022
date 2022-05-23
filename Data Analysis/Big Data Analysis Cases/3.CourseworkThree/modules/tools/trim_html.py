def trim_html(html):
    html = html.split("<body>")[-1]
    html = html.split("</body>")[0]
    return(html)
import os

from flask import Flask, render_template, request
import plotly
plotly.tools.set_credentials_file(username='abhati01', api_key='dbYmCxlRtPhX1MnoCsv7')
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
import json
from collections import defaultdict
import os
import traceback

app = Flask(__name__, static_url_path='/static')

def fetch_data():
    """ read all json files from a directory and create default dict lookup """
    location_lookup = defaultdict(lambda: defaultdict(lambda: 0))
    for json_file in os.listdir('app_data'):
        if json_file.endswith('.json'):
            path = os.path.join(os.getcwd(), 'app_data', json_file)
            with open(path, 'rb') as f:
                tweets = (json.loads(tweet) for tweet in f.readlines())
                for tweet in tweets:
                    try:
                        name = tweet['place']
                        if name not in location_lookup.keys() and tweet['sentiment'] == 0:
                            location_lookup[name]['zero'] = 1
                        elif name not in location_lookup.keys() and tweet['sentiment'] == 1:
                            location_lookup[name]['one'] = 1
                        elif name not in location_lookup.keys() and tweet['sentiment'] == 2:
                            location_lookup[name]['two'] = 1
                        elif name in location_lookup.keys() and tweet['sentiment'] == 0:
                            location_lookup[name]['zero'] += 1
                        elif name in location_lookup.keys() and tweet['sentiment'] == 1:
                            location_lookup[name]['one'] += 1
                        else:
                            location_lookup[name]['two'] += 1
                    except Exception as e:
                        print(traceback.format_exc())
                        # print(e)
                        continue
    filter_lookup = {k:v for k,v in location_lookup.items() if (v['zero']+v['one']+v['two'] > 5)}
    return filter_lookup


@app.route('/submitted', methods=['POST'])
def graph():
    total_positive=0
    total_negative=0
    total_neutral=0
    place=request.form.get("countries")
    my_data=fetch_data()
    countries=[key for key,value in my_data.items() if (my_data[key]['zero'])]
    labels =['Positive','Negative','Neutral']
    if(place=='Show all'):
        for key, value in my_data.items():
            total_positive+=my_data[key]['zero']
            total_negative+=my_data[key]['one']
            total_neutral+=my_data[key]['two']
        values=[total_positive,total_negative,total_neutral]
    else:
        values=[my_data[place]['zero'],my_data[place]['one'],my_data[place]['two']]
    trace=go.Pie(labels=labels,values=values, hole = 0.4)
    data = [trace]
    layout = dict(title = place, titlefont = dict(family = 'monospace',size=28, color='black'),  paper_bgcolor= "rgba(0,0,0,0)")
    fig = dict(data=data, layout=layout)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('index.html',graphJSON=graphJSON, countries=countries)

@app.route('/')
def line():
    my_data=fetch_data()
    countries=list(my_data.keys())
    return render_template('index.html',countries=countries)


if __name__ == "__main__":
    app.run()



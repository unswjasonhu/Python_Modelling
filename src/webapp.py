import time

from flask import Flask
from flask import render_template, request, jsonify
import re
import urllib
from config import config

import os
import sys
import inspect

# Add folder to path 
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))

if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from app.services.services import get_estimates_data_service, generate_2d_plot

app = Flask(__name__, static_url_path='')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dev')
def dev():
    return render_template('index_dev.html')


@app.route('/get_estimates_data', methods=['GET'])
def get_estimates_data():
    input_datetime = request.args.get('input_datetime')
    input_date = request.args.get('input_date')

    try:
        lat = request.args.get('lat')
        lon = request.args.get('lon')
        if lat or lon:
            lat = float(request.args.get('lat'))
            lon = float(request.args.get('lon'))
    except ValueError:
        return jsonify({'error': 'Malformed lat or lon given given'})

    body = []
    if input_datetime:
        try:
            time.strptime(input_datetime, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return jsonify({'error': 'Invalid input_datetime given'})
        else:
            body = get_estimates_data_service(input_datetime=input_datetime,
                                              input_date=input_date,
                                              lat=lat,
                                              lon=lon)

    if input_date:
        try:
            time.strptime(input_date, "%Y-%m-%d")
        except ValueError:
            return jsonify({'error': 'Invalid input_date given'})
        else:
            body = get_estimates_data_service(input_datetime=input_datetime,
                                              input_date=input_date,
                                              lat=lat,
                                              lon=lon)

    return jsonify(body)


@app.route('/generate_plot', methods=['GET'])
def generate_plot():
    input_datetime = request.args.get('input_datetime')

    try:
        time.strptime(input_datetime, "%Y-%m-%d %H:%M:%S")
        # can use create create_heatmap to generate an image
        plot_name = generate_2d_plot(input_datetime)
    except ValueError:
        return jsonify({'error': 'Invalid input_datetime given'})
    plot_name = "{}.png".format(plot_name)

    try:
        url = re.search(r'static.*png', plot_name).group()
    except:
        return jsonify({'error': 'Invalid output returned'})

    url = urllib.quote('/modeling/' + url)
    return jsonify({'success': url})

if __name__ == '__main__':
    app.debug = True

    app.config.from_object(config)
    print('App variables are:', app.__dict__)
    if app.debug:
        app.run(debug=True, host='0.0.0.0')
    else:
        app.run(debug=True)

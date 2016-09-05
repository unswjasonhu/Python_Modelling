import time

from flask import Flask
from flask import render_template, request, jsonify
import re
import urllib

from services.services import get_estimates_data_service, generate_2d_plot

app = Flask(__name__, static_url_path='')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dev')
def dev():
    return render_template('index_dev.html')

@app.route('/get_estimates_data', methods = ['GET'])
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

    #lat = -33.92313
    #lon = 150.98812

    #print input_datetime
    #input_datetime = "2015-09-03 10:00:00"

    body=[]
    if input_datetime:
        try:
            time.strptime(input_datetime, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return jsonify({'error': 'Invalid input_datetime given'})
        else:
            body = get_estimates_data_service(input_datetime=input_datetime, input_date=input_date, lat=lat, lon=lon)

    #input_date = "2015-08-05"
    if input_date:
        try:
            time.strptime(input_date, "%Y-%m-%d")
        except ValueError:
            return jsonify({'error': 'Invalid input_date given'})
        else:
            body = get_estimates_data_service(input_datetime=input_datetime, input_date=input_date, lat=lat, lon=lon)

    return jsonify(body)


@app.route('/generate_plot', methods= ['GET'])
def generate_plot():
    input_datetime = request.args.get('input_datetime')

    try:
        time.strptime(input_datetime, "%Y-%m-%d %H:%M:%S")
        #can use create create_heatmap to generate an image
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

app.debug=False

if __name__ == '__main__':
    app.run(debug=False)


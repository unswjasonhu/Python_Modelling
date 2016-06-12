import time

from flask import Flask
from flask import render_template, request, jsonify, send_from_directory

from services.get_estimates_data import get_estimates_data_service

app = Flask(__name__, static_url_path='')

#@app.route('/')
#def index():
#    return "hello world"
#    #return render_template('index.html')

@app.route('/js/<path:path>')
def send_js(path):
        return send_from_directory('js', path)

@app.route('/')
def index():
    #return "hello modelling index"
    return render_template('index.html')

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
    except:
        return jsonify({'error': 'Malformed lat or lon given given'})

    #lat = -33.92313
    #lon = 150.98812

    #print input_datetime
    #input_datetime = "2015-09-03 10:00:00"

    body=[]
    if input_datetime:
        try:
            time.strptime(input_datetime, "%Y-%m-%d %H:%M:%S")
            body = get_estimates_data_service(input_datetime=input_datetime, input_date=input_date, lat=lat, lon=lon)
        except:
            return jsonify({'error': 'Invalid input_datetime given'})

    #input_date = "2015-08-05"
    if input_date:
        try:
            time.strptime(input_date, "%Y-%m-%d")
            body = get_estimates_data_service(input_datetime=input_datetime, input_date=input_date, lat=lat, lon=lon)
        except:
            return jsonify({'error': 'Invalid input_date given'})

    return jsonify(body)

app.debug=True

if __name__ == '__main__':
    app.run(debug=True)


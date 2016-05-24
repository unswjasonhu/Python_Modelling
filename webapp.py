import time

from flask import Flask
from flask import render_template, request, jsonify

from services.get_estimates_data import get_estimates_data_service

app = Flask(__name__)

#@app.route('/')
#def index():
#    return "hello world"
#    #return render_template('index.html')

@app.route('/')
def index():
    #return "hello modelling index"
    return render_template('index.html')

@app.route('/get_estimates_data', methods = ['GET'])
def get_estimates_data():
    #input_datetime = request.form.get('date_time')
    #input_date = request.form.get('date')
    lat = float(request.args.get('lat'))
    lon = float(request.args.get('lon'))
    #lat = -33.92313
    #lon = 150.98812

    #print input_datetime
    input_datetime = "2015-09-03 10:00:00"
    
    try:
        time.strptime(input_datetime, "%Y-%m-%d %H:%M:%S")
    except:
        pass

    input_date = "2015-08-05"
    try:
        time.strptime(input_date, "%Y-%m-%d")
    except:
        return jsonify({'error': 'Invalid datetime given'})

    body = get_estimates_data_service(input_datetime, input_date, lat, lon)
    return jsonify(body)

app.debug=True

if __name__ == '__main__':
    app.run(debug=True)


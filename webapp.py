from flask import Flask
from flask import render_template

app = Flask(__name__)

#@app.route('/')
#def index():
#    return "hello world"
#    #return render_template('index.html')

@app.route('/')
def hello_world():
    #return "hello modelling index"
    return render_template('index.html')

#@app.route('/hari')
#def hari():
#    return 'Hello Hari!'

app.debug=True

if __name__ == '__main__':
    app.run(debug=True)


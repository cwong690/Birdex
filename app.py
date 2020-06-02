from flask import Flask, render_template, request, jsonify 
import pickle
import numpy as np

from predict import predict
from test_script import test_script

from pymongo import MongoClient


# Create app
app = Flask(__name__)

# Render home page
@app.route('/',methods=['GET'])
def home():

    # connect to database
    client = MongoClient('localhost', 27017)
    db = client['frauds']
    table = db['new_events12']
    
    events = table.find().sort([('sequence', -1)]).sort([('fraud_probability', -1)]).limit(50)
    legend = 'Risk Level'
    entries = table.find().sort([('sequence', -1)]).sort([('fraud_probability', -1)]).limit(1000)
    unable, low, medium, high= 0, 0, 0, 0
    for each_ in entries:
        if each_['risk_factor'] == 'Unable to Predict':
            unable += 1
        elif each_['risk_factor'] == 'low':
            low += 1
        elif each_['risk_factor'] == 'medium':
            medium += 1
        else:
            high += 1
    labels = ['unable','low', 'medium', 'high']

    # render the template and pass the events
    return render_template('home.html', data=events, unable=unable, low=low, medium=medium, high=high, labels=labels, legend=legend)

# Execute logic of prediction
@app.route('/score', methods=['POST'])
def score():
    pass



if __name__ == '__main__':
    # load saved pickled model
    with open('models/grad_boost_model.p', 'rb') as mod:
        model = pickle.load(mod)

    # connect to database
    client = MongoClient('localhost', 27017)
    db = client['frauds']
    table = db['new_events12']
    
    # run flask app
    app.run(host='0.0.0.0', port=8000, debug=True)
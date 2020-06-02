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
    
    probs = table.find().sort([('sequence', -1)]).sort([('fraud_probability', -1)]).limit(1000)
    
    probas = []
    for p in probs:
        probas.append(p['fraud_probability'])

    # render the template and pass the events
    return render_template('second_home.html', data=events, probas=probas, unable=unable, low=low, medium=medium, high=high, labels=labels, legend=legend)

# Execute logic of prediction
@app.route('/score', methods=['POST'])
def score():
    pass

# @app.route("/line_chart")
# def line_chart():
#     legend = 'Temperatures'
#     temperatures = [73.7, 73.4, 73.8, 72.8, 68.7, 65.2,
#                     61.8, 58.7, 58.2, 58.3, 60.5, 65.7,
#                     70.2, 71.4, 71.2, 70.9, 71.3, 71.1]
#     times = ['12:00PM', '12:10PM', '12:20PM', '12:30PM', '12:40PM', '12:50PM',
#              '1:00PM', '1:10PM', '1:20PM', '1:30PM', '1:40PM', '1:50PM',
#              '2:00PM', '2:10PM', '2:20PM', '2:30PM', '2:40PM', '2:50PM']
#     return render_template('line_chart.html', values=temperatures, labels=times, legend=legend)



if __name__ == '__main__':
    # load saved pickled model
    with open('models/grad_boost_model.p', 'rb') as mod:
        model = pickle.load(mod)

    # connect to database
    client = MongoClient('localhost', 27017)
    db = client['frauds']
    table = db['new_events12']
    
    # run flask app
    app.run(host='0.0.0.0', port=8080, debug=True)
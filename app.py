from flask import Flask, flash, render_template, request, redirect, jsonify, url_for
import pickle
import numpy as np

# from predict import predict
# from test_script import test_script

from pymongo import MongoClient


# Create app
app = Flask(__name__)

# Render home page
@app.route('/',methods=['GET', 'POST'])
def home():

    # # connect to database
    # client = MongoClient('localhost', 27017)
    # db = client['frauds']
    # table = db['new_events12']
    
    if request.method == 'GET':
        # show the upload form
        return render_template('home.html')

    if request.method == 'POST':
        # TODO: Logic to process uploaded images

        passed = True # for now
        if passed:
            return redirect(url_for('predict', filename='test.png'))
        else:
            return redirect(url_for('error'))

@app.route('/predict/<filename>', methods=['GET'])
def predict(filename):
    # TODO: Logic to load the uploaded image filename and predict the
    # labels

    return render_template('predict.html')


@app.errorhandler(500)
def server_error(error):
    return render_template('error.html'), 500







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
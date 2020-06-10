from flask import Flask, flash, render_template, request, redirect, jsonify, url_for
import pickle
import pandas as pd
import numpy as np
import os
from werkzeug import secure_filename

from PIL import Image
import requests
from io import BytesIO

import pickle

import tensorflow as tf
from tensorflow.keras.models import load_model
# load_xception = load_model('saved_models/xception_final.h5')
# orders_xception = load_model('saved_models/orders_xception.h5')
species_xception = load_model('saved_models/species_xception.h5')

final_orders = pd.read_csv('data/final_orders.csv', index_col=0)

# from predict import predict
# from test_script import test_script

# from pymongo import MongoClient


# Create app
app = Flask(__name__)

# Choose configuration, env is development or production
app.config.from_object('config.DevConfig')

# make sure users are only allowed to upload image files
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return ('.' in filename) and (filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

def model_predict(filepath):
    img = Image.open(filepath)
    img_rs = np.array(img.resize((299,299)))/255
    prediction = species_xception.predict(img_rs.reshape(1,299,299,3))
    return np.round(prediction * 100, 1)[0]

# Render home page
@app.route('/', methods=['GET', 'POST'])
def home():
#     if request.method == 'GET':
    return render_template('home.html')
    
@app.route('/chart', methods=['GET'])
def chart():
    species_sort = final_orders.groupby('species_group').count()['family'].sort_values(ascending=False)
    species_count = species_sort.values
    species_index = species_sort.index.values
    
    order_sort = final_orders.groupby('order').count()['species_group'].sort_values(ascending=False)
    order_count = order_sort.values
    order_index = order_sort.index.values
    
    return render_template('chart.html', 
                           species_count=species_count, 
                           species_index=species_index, 
                           order_count=order_count, 
                           order_index=order_index)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
#         check if the post request has the file part
        if 'image' not in request.files:
            flash(f'No image: {request.files}')
            return redirect(request.url)
        
        file = request.files['image']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash(f'No selected file: {file.filename}')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
#             flash(f'{filename} saved successfully!')
            labels = np.unique(np.array(final_orders['species_group'].values))
            prediction = model_predict(filepath)
            top_3 = prediction.argsort()[-1:-4:-1]
            return render_template('predict.html', prediction=prediction, labels=labels, top_3=top_3, filepath=filepath)
        else:
            flash('An error occurred, try again.')
            return redirect(request.url)

# response = requests.get(url)
# img = Image.open(BytesIO(response.content))

@app.errorhandler(500)
def server_error(error):
    return render_template('error.html'), 500



if __name__ == '__main__':
    # connect to database
    # client = MongoClient('localhost', 27017)
    # db = client['frauds']
    # table = db['new_events12']
    
    # run flask app
    app.run(host='0.0.0.0', port=8000, debug=True)
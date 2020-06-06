from flask import Flask, flash, render_template, request, redirect, jsonify, url_for
import pickle
import numpy as np
import os
from werkzeug import secure_filename

from PIL import Image
import requests
from io import BytesIO

import sys
sys.path.append("../")

from tensorflow.keras.models import load_model
load_xception = load_model('saved_models/xception_final.h5')

# from predict import predict
# from test_script import test_script

# from pymongo import MongoClient


# Create app
app = Flask(__name__)
app.config.from_object('config.DevConfig')

# make sure users are only allowed to upload image files
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return ('.' in filename) and (filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

def predict(filepath):
    img = Image.open(filepath)
    img_rs = np.array(img.resize((299,299)))/255
    prediction = load_xception.predict(img_rs.reshape(1,299,299,3))
    return np.round(prediction * 100, 1)

# Render home page
@app.route('/',methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        # show the upload form
        return render_template('home.html')
    
    if request.method == 'POST':
        # check if the post request has the file part
        if 'image' not in request.files:
            flash(f'No image part: {request.files}')
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
            flash(f'{filename} saved successfully!')
            
            labels = ['Duck', 'Finch', 'Hawk']
            label_range = range(len(labels))
            prediction = predict(filepath)
            return render_template('home.html', prediction=prediction, labels=labels, label_range=label_range)
        else:
            flash('An error occurred, try again.')
            return redirect(request.url)


# response = requests.get(url)
# img = Image.open(BytesIO(response.content))

# @app.route('/predict/<filepath>', methods=['GET'])
# def predict(filepath):
    
#     img_bytes = BytesIO(filepath)
#     open_img = Image.open(img_bytes)
#     arr = np.array(open_img.resize((299,299)))
#     print(arr)

#     return render_template('predict.html')


@app.errorhandler(500)
def server_error(error):
    return render_template('error.html'), 500



if __name__ == '__main__':
    # load saved pickled model
   # with open('models/grad_boost_model.p', 'rb') as mod:
       # model = pickle.load(mod)

    # connect to database
    # client = MongoClient('localhost', 27017)
    # db = client['frauds']
    # table = db['new_events12']
    
    # run flask app
    app.run(host='0.0.0.0', port=8000, debug=True)
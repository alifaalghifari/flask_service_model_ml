from flask import (Flask, flash, make_response,
                   redirect, render_template, request, session, url_for, jsonify
                   )
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)


app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
# app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'uploads'

model = load_model('model_ml/my_model.h5')

@app.route('/')
def index():
    return render_template('sendImage.html')

@app.route('/resultmodel', methods=['POST'])
def result_model():
        
        images = request.files['img']

        # save file
        if images.filename != '':
            images.save(os.path.join(
                app.config['UPLOAD_PATH'], images.filename))

        img = tf.keras.preprocessing.image.load_img(
            os.path.join(
                app.config['UPLOAD_PATH'], images.filename), target_size=(224, 224))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        img = np.vstack([x])
        classes = model.predict(img, batch_size=10)
        result = np.argmax(classes[0])

        # menghapus file
        os.remove(os.path.join(
            app.config['UPLOAD_PATH'], images.filename))

        # ['LeafBlast', 'Healthy', 'BrownSpot', 'Hispa']
        if result == 0:
            name = 'LeafBlast'
        elif result == 1:
            name = 'Healthy'
        elif result == 1:
            name = 'BrownSpot'
        else:
            name = 'Hispa'

        # return json 
        # response_json = {
        #     name : name
        # }
        # return jsonify(response_json)

        # return web
        return render_template('resultModel.html', training=str(classes), hasil=str(result), nama=name )
if __name__ == '__main__':
    app.run(debug=True)

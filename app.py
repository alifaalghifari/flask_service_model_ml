from flask import (Flask,  render_template, request
                   )
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os
import pickle
import pandas as pd

app = Flask(__name__)


app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
# app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'uploads'

model = load_model('model_ml/my_model.h5')

@app.route('/')
def index():
    return render_template('sendImage.html')

df = pickle.load(open("model_ml/recommend_data.pkl", "rb"))
similarity = pickle.load(open("model_ml/similarity.pkl", "rb"))
# list_movie = np.array(df["cleaned_desc"])


def recommend_pestisida():

    # nama = request.form['nama']
    # jenis = request.form['jenis']
    # data = df.loc[df['kegunaan'] == jenis]
    # data.reset_index(level=0, inplace=True)

    indices = pd.Series(df.index, index=df['nama'])

    # Get the pairwsie similarity scores
    idx = indices['Filia 525Se 250Ml Obat Hawar Daun Dan Blast Original']
    # return idx
    # return idx
    # print(idx)
    sig = list(enumerate(similarity[idx]))  # Sort the names
    # return sig
    # Scores of the 5 most similar books
    sig = sorted(sig, key=lambda x: x[1], reverse=True)
    # return sig

    sig = sig[1:10]  # Book indicies
    tourist_indices = [i[0] for i in sig]

    #   # Top 5 tourist recommendation
    rec = df[['nama', 'kegunaan', 'tempat',
              'berat', 'image-src', 'product_link']].iloc[tourist_indices]

    # print(rec)
    return rec

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
        elif result == 2:
            name = 'BrownSpot'
        else:
            name = 'Hispa' 
            
        # get pesticide recommend
        rec = recommend_pestisida()

        # return json 
        # response_json = {
        #     name : name,
        #     recommendations : rec
        # }
        # return jsonify(response_json)

       
        # return web
        return render_template('resultModel.html', training=str(classes), hasil=str(result), nama=name,recommend=rec )
if __name__ == '__main__':
    app.run(debug=True)

import os
from io import BytesIO

from flask import Flask, render_template, request
from werkzeug import secure_filename
from PIL import Image
import numpy as np
from keras.models import load_model
import tensorflow as tf

graph = tf.get_default_graph()
with graph.as_default():
    model = load_model('inception_v3_guitar_classifier.h5')

app = Flask(__name__)

class2idx = {
    'B.C. Rich': 0,
    'Caparison': 1,
    'DEAN': 2,
    'ESP': 3,
    'Fender': 4,
    'Gibson': 5,
    'Ibanez': 6,
    'Jackson': 7,
    'Kiesel': 8,
    'Mayones': 9,
    'Paul Reed Smith': 10,
    'SCHECTER': 11,
    'Strandberg': 12,
    'Suhr': 13
}

idx2class = {v: k for k, v in class2idx.items()}


def predict(stream):
    bytes = bytearray(stream.read())
    img = Image.open(BytesIO(bytes))
    img_resize = img.resize((224, 224))
    img_array = np.asarray(img_resize)
    img_array = img_array / 255.
    img_array = img_array.reshape((1, 224, 224, 3))
    with graph.as_default():
        pred_idx = np.argmax(model.predict(img_array))
    pred_brand = idx2class[pred_idx]

    return pred_brand


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == "POST":
        img_file = request.files['img_file']
        if img_file:
            pred_brand = predict(img_file.stream)
            return render_template('index.html', pred_brand=pred_brand)
        else:
            return render_template('index.html', warning="画像を選択してください!")


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8090)

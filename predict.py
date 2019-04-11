from PIL import Image
import numpy as np
from keras.models import load_model
import tensorflow as tf

graph = tf.get_default_graph()


def predict(img_url):
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
    with graph.as_default():
        model = load_model('inceptionv3_guitar_classifier.h5')
        img = Image.open(img_url)
        img_resize = img.resize((224, 224))
        img_array = np.asarray(img_resize)
        img_array = img_array / 255.
        img_array = img_array.reshape((1, 224, 224, 3))

        pred_idx = np.argmax(model.predict(img_array))

        pred_brand = idx2class[pred_idx]

    return pred_brand

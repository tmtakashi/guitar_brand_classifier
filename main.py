import os

from flask import Flask, render_template, request
from werkzeug import secure_filename
from predict import predict

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == "POST":
        img_file = request.files['img_file']
        if img_file:
            filename = secure_filename(img_file.filename)
            img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img_url = 'uploads/' + filename

            pred_brand = predict(img_url)

            return render_template('index.html', img_url=img_url, pred_brand=pred_brand)


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8090)

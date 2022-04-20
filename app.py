from __future__ import division, print_function

# Flask utils
import os

from flask import Flask, redirect, url_for, request, render_template

# Define a flask app
from werkzeug.utils import secure_filename

from prediction import model_predict

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(
            base_path, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path)
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

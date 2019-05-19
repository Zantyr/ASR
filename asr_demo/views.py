from flask import render_template, flash, redirect

from asr_demo import app

import os
import scipy.io.wavfile as sio

from engine import get_engine

@app.route('/')
def index():
    app.logger.warning('sample message')
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'recording' not in request.files:
        return json.dumps(['No file passed'])
    file = request.files['recording']
    if file.filename == '':
        return json.dumps(['No file passed'])
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        sr, recording = sio.read(filename)[0]
        assert sr == 16000
        engine = get_engine(recording)
        phonemes = engine.predict(recording)
    else:
        return json.dumps(['No file passed'])

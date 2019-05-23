from flask import render_template, redirect, request

from asr_demo import app

import json
import numpy as np
import os
import scipy.io.wavfile as sio
import tempfile

from asr_demo.engine import get_engine

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print(request.files)
    if 'recording' not in request.files:
        json.dumps(['No file passed']), 400
    file = request.files['recording']
    if file.filename == '':
        json.dumps(['No file passed']), 400
    if file:
        filename = tempfile.mktemp()
        try:
            file.save(filename)
            rec = sio.read(filename)[1].astype(np.float32) / 2**15
            engine = get_engine()
            return engine.predict_phones(rec)
        finally:
            os.remove(filename)
    else:
        return json.dumps(['No file passed']), 400 

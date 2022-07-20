from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import nibabel as nib
import numpy as np
import tensorflow as tf
from flask import send_file
model = tf.keras.models.load_model('LiverModelv5')

app = Flask(__name__)

imgid = 0
img_path = './images'
if not os.path.isdir(img_path): 
    os.mkdir(img_path)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/upload",methods=['GET','POST'])
def upload():
   if request.method == 'POST':
        img = request.files['file']
        if not img:
            return 'No Images Found'
        elif img.filename.endswith('.nii') or img.filename.endswith('.nii.tar.gz'):
            img.save(os.path.join(img_path,secure_filename(img.filename)))
            img_data = nib.load(os.path.join(img_path,secure_filename(img.filename)))
            print("request")
            # if request.files['mask']:
            #     mask = request.files['mask']
            #     mask.save(os.path.join(img_path,secure_filename(f.filename)))
            #     mask_data = nib.load(os.path.join(img_path,secure_filename(mask.filename)))
            imgf_data = img_data.get_fdata()
            imgf_data = imgf_data.reshape((-1,512,512))
            print(imgf_data.shape)
            result = model.predict(imgf_data)
            print(result)
            result = result.reshape((512,512,-1))
            nib.save(array_img, f'{img.filename}_masked.nii')
            return send_file('./f{img.filename}_masked.nii', attachment_filename='./f{img.filename}_masked.nii')
            # return 'file uploaded successfully'
        else:
            print(f.filename)
            return 'Wrong File type'
        
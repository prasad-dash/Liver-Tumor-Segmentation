from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

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
        f = request.files['file']
        if not f:
            return 'No Images Found'
        elif f.filename.endswith('.jpg') or f.filename.endswith('.png'):
            f.save(os.path.join(img_path,secure_filename(f.filename)))
            return 'file uploaded successfully'
        else:
            print(f.filename)
            return 'Wrong File type'
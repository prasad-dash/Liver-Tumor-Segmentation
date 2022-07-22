# flask imports
from flask import Flask, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
import uuid  # for public id
from flask import send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from flask import send_file
# imports for PyJWT authentication
import jwt
from datetime import datetime, timedelta
from functools import wraps
from werkzeug.utils import secure_filename
from flask_cors import CORS

import tensorflow as tf
import numpy as np
import nibabel as nib
import os 
import pickle
# creates Flask object
app = Flask(__name__)
CORS(app)

# configuration
# NEVER HARDCODE YOUR CONFIGURATION IN YOUR CODE
# INSTEAD CREATE A .env FILE AND STORE IN IT
app.config["SECRET_KEY"] = "tnNLGgVpJFN66THS0lxJtw"
app.config['CORS_HEADERS'] = 'Content-Type'

LIST_OF_USERS = []
if os.path.exists('parrot.pkl'):
    with open('parrot.pkl', 'rb') as f:
        LIST_OF_USERS = pickle.load(f)


model = tf.keras.models.load_model('LiverModelv5')

# decorator for verifying the JWT
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        # jwt is passed in the request header

        auth_header = request.headers.get('Authorization')
        if auth_header:
            token = auth_header.split(" ")[1]
        

        # return 401 if token is not passed
        if not token:
            return jsonify({"message": "Token is missing !!"}), 401

        try:
            # decoding the payload to fetch the stored details
            data = jwt.decode(token, app.config["SECRET_KEY"])
            current_user = [
                i
                for i in LIST_OF_USERS
                if i["firstName"] == data["firstName"]
                and i["lastName"] == data["lastName"]
                and i["username"] == data["username"]
            ][0]

        except Exception as exe:
            print(exe)
            return jsonify({"message": "Token is invalid !!"}), 401
        # returns the current logged in users contex to the routes
        return f(current_user, *args, **kwargs)

    return decorated


# User Database Route
# this route sends back list of users
@app.route("/user", methods=["GET"])
@token_required
def get_all_users(current_user):
    return jsonify({"users": LIST_OF_USERS})


# route for logging user in
@app.route("/login", methods=["POST"])
def login():
    # creates dictionary of form data
    auth = request.form

    if not auth or not auth.get("username") or not auth.get("password"):
        # returns 401 if any username or / and password is missing
        return make_response(
            "Could not verify",
            401,
            {"WWW-Authenticate": 'Basic realm ="Login required !!"'},
        )
    user = [i for i in LIST_OF_USERS if i["username"] == auth.get("username")]

    if len(user) == 0:
        # returns 401 if user does not exist
        return make_response(
            "Could not verify",
            401,
            {"WWW-Authenticate": 'Basic realm ="User does not exist !!"'},
        )

    if check_password_hash(user[0]['password'],(auth.get("password")+user[0]['firstName'])):
        # generates the JWT Token
        token = jwt.encode(
            {
                "firstName": user[0]["firstName"],
                "lastName": user[0]["lastName"],
                "username": user[0]["username"],
            },
            app.config["SECRET_KEY"],
        )

        return make_response(jsonify({
                "token": token.decode("UTF-8"),
                "user":{
                    "firstName": user[0]["firstName"],
                    "lastName": user[0]["lastName"],
                    "username": user[0]["username"],
                    "isAdmin": False
                    },
                }), 201)
    # returns 403 if password is wrong
    return make_response(
        "Could not verify",
        403,
        {"WWW-Authenticate": 'Basic realm ="Wrong Password !!"'},
    )


# signup route
@app.route("/signup", methods=["POST"])
def signup():
    # creates a dictionary of the form data
    data = request.json

    # gets name, username and password
    firstName, lastName, username = (
        data.get("firstName"),
        data.get("lastName"),
        data.get("username"),
    )
    password = data.get("password")

    # checking for existing user
    user = [
        i
        for i in LIST_OF_USERS
        if i["firstName"] == firstName
        and i["lastName"] == lastName
        and i["username"] == username
    ]
    if len(user) == 0:
        LIST_OF_USERS.append(
            {
                "firstName": firstName,
                "lastName": lastName,
                "username": username,
                "password": generate_password_hash(password+firstName),
            }
        )
        token = jwt.encode(
            {
                "firstName": firstName,
                "lastName": lastName,
                "username": username
            },
            app.config["SECRET_KEY"],
        )
        with open('parrot.pkl', 'wb') as f:
            pickle.dump(LIST_OF_USERS, f)
        return make_response(jsonify({
                "token": token.decode("UTF-8"),
                "user":{
                    "firstName":firstName,
                    "lastName": lastName,
                    "username": username,
                    "isAdmin": False
                    },
                }), 200)
    else:
        # returns 202 if user already exists
        return make_response(jsonify({"message":"User Already Exists"}), 400)


@app.route("/upload", methods=["POST"])
@token_required
def upload(current_user=None):
    f = request.files["file"]
    f.save(secure_filename(f.filename))
    print("Data Saved")
    if request.form.get('evaluate')=="true":
        mask = request.files["mask"]
        mask.save(secure_filename(mask.filename))
        print("Mask Saved")
        mask_img = nib.load(mask.filename)
        mask_img_data = mask_img.get_fdata()
        mask_img_data = mask_img_data.reshape(-1,512,512)

    img = nib.load(f.filename)
    img_data = img.get_fdata()
    img_data = img_data.reshape(-1,512,512)
    if request.form.get('evaluate')=="true":
        results = model.evaluate(img_data,mask_img_data)
        return jsonify({"results":results})
        
    res = model.predict(img_data)
    res = res.reshape(512,512,-1)
    img = nib.Nifti1Image(res, np.eye(4))
    img.get_data_dtype() == np.dtype(np.int16)
    nib.save(img,  'res.nii.gz')  
    return send_from_directory('.','res.nii.gz', as_attachment=True,filename='res.nii.gz')


if __name__ == "__main__":
    # setting debug to True enables hot reload
    # and also provides a debugger shell
    # if you hit an error while running the server
    app.run(debug=True)

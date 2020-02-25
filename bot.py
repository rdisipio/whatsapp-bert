from flask import Flask, flash, request, jsonify, send_file, redirect, render_template
from werkzeug.utils import secure_filename

import requests
import os
#from tokenizer import Tokenizer
#from intent_classifier import IntentClassifier

app = Flask(__name__)
#app.config['SERVER_NAME'] = "127.0.0.1:4996"

app.config['UPLOAD_FOLDER'] = "/Users/Riccardo/Desktop/uploads"

@app.route('/', methods=['GET'])
def hello():
   return 'Hello, World!'


@app.route('/bot', methods=['POST', 'GET'])
def bot():
    response = None
    
    if request.method == 'GET':
        req = request.get_json()
        print(req)
        message = req['message']
        print(message)
        if 'quote' == message:
            # return a quote
            r = requests.get('https://api.quotable.io/random')
            if r.status_code == 200:
                data = r.json()
                quote = f'{data["content"]} ({data["author"]})'
            else:
                quote = 'I could not retrieve a quote at this time, sorry.'
            response = quote + '\n'
    elif request.method == 'POST':
        print("Receiving file")
        if request.mimetype == 'application/pdf':
            print(request.files)
            if not request.files:
                response = 'No file part'
            doc = request.files["image"]
            filename = secure_filename(doc.filename)
            print(filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            doc.save(filepath)
            response = "Receiving a PDF file"
    print(response)
    return response

@app.route('/send_resume', methods=['POST'])
def send_resume():
    print("Receiving file")
    if request.method == 'POST':
        print(request.mimetype)
        f = request.files['resume']
        filename = "{}/{}".format(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        print(filename)
        f.save(filename)

    response = "Got file: {}".format(filename)
    return response


if __name__ == "__main__":
    app.run(debug=True)

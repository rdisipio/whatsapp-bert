from flask import Flask, request
import requests
from twilio.twiml.messaging_response import MessagingResponse

from tokenizer import Tokenizer
from intent_classifier import IntentClassifier

app = Flask(__name__)
#app.config['SERVER_NAME'] = "127.0.0.1:4996"

@app.route('/', methods=['GET'])
def hello():
   return 'Hello, World!'


@app.route('/bot', methods=['POST'])
def bot():
    incoming_msg = request.values.get('Body', '').lower()
    resp = MessagingResponse()
    msg = resp.message()
    responded = False
    if 'quote' in incoming_msg:
        # return a quote
        r = requests.get('https://api.quotable.io/random')
        if r.status_code == 200:
            data = r.json()
            quote = f'{data["content"]} ({data["author"]})'
        else:
            quote = 'I could not retrieve a quote at this time, sorry.'
        msg.body(quote)
        responded = True
    if 'cat' in incoming_msg:
        # return a cat pic
        msg.media('https://cataas.com/cat')
        responded = True
    if not responded:
        msg.body('I only know about famous quotes and cats, sorry!')
    print(str(resp))
    return str(resp)

if __name__ == "__main__":
    app.run(debug=True)
